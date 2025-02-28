"""
This file contains a script to export labels from a Labelbox project and to extract the single video frames,
landmark points, and area feature masks from the exported labels.

Changes:
    - If no mask is found for a frame, we create an all-black mask for that frame.
    - If a bolus mask is corrupt or cannot be fetched or cannot be merged, we write an all-white image (will be removed later)
    - We now read the API key from a file named 'apikey.txt' in the same directory as this script.
    - We now suppress deprecation warnings
"""

import labelbox as lb
import ndjson
import os
import shutil
import cv2
import requests
from io import BytesIO
import numpy as np
import tempfile
import requests
import argparse
from typing import List, Dict
import csv
from tqdm import tqdm

# surpress deprecation warnings
import warnings
warnings.filterwarnings("ignore")



def export_labels(project, output_directory):
    """
    Exports labels marked as 'DONE' from a Labelbox project and saves them as
    '<output_directory>/exported_data.ndjson'.

    Parameters:
    - project (labelbox.schema.project.Project): Labelbox project from which the labels are exported
    - output_directory (str): directory where the exported data is saved as ndjson file

    Returns:
    - labels (list): list containing the exported labels
    """

    # Start the export of the labels
    task = project.export_v2(
        params={
            "data_row_details": True,
            "metadata_fields": True,
            "attachments": True,
            "project_details": True,
            "performance_details": True,
            "label_details": True,
            "interpolated_frames": True,
        }
    )

    # Wait until the export is finished
    try:
        task.wait_till_done()

        # Check if any errors occured during the export
        if task.errors:
            print(f"An error occurred while trying to export the labels: {task.errors}")
            return None

        # Access the result if no errors occured
        labels = task.result

    # Handle any exception that might have occurred during the label export
    except Exception as e:
        print(f"An error occurred while trying to export the labels: {e}")
        return None

    # Only keep the labels which are marked as "DONE"
    labels = [dr for dr in labels if dr["projects"][project.uid]["project_details"]["workflow_status"] == "DONE"]

    # Save the exported labels as '<output_directory>/exported_data.ndjson'
    output_file_path = os.path.join(output_directory, "exported_data.ndjson")
    with open(output_file_path, "w") as output_file:
        ndjson.dump(labels, output_file)

    return labels

def export_frames_and_bolus_masks_skip_on_error(
    labels,
    frames_output_dir,
    masks_output_dir,
    client,
    csv_path,
    start_index=0,
    end_index=None
):
    """
    For each labeled video in `labels`:
      - Download the video via its URL (WITHOUT passing client.headers, to avoid 401).
      - Read ALL frames using OpenCV.
      - For each frame:
         1) Lookup Bolus annotations for that local frame index (if any).
         2) Attempt to fetch/merge Bolus masks.
             -> If ANY error, skip this frame (no frame .png, no mask .png).
         3) If no bolus is annotated, produce an all-black mask.
         4) If everything is fine, save the frame & mask with a consistent global_frame_idx.
      - We record each SUCCESSFUL frame in `data_overview.csv`.

    We'll print a summary at the end with total frames read, total frames saved,
    total frames skipped. The resulting CSV has 3 columns:
    [frame_idx, video_name, local_frame_idx].

    Args:
        labels (List[Dict]): The exported Labelbox data
        frames_output_dir (str): Where to save frames
        masks_output_dir (str): Where to save masks
        client: Labelbox client for authenticated requests to mask URLs
        csv_path (str): Where to write the final data_overview.csv
        start_index (int), end_index (int): subset of labels to process
    """

    if end_index is None:
        end_index = len(labels)

    # Prepare for CSV logging
    data_overview = []
    csv_header = ["frame_idx", "video_name", "frame"]

    # Some counters for summary
    total_videos = 0
    total_frames_read = 0
    saved_frames_count = 0
    skipped_frames_count = 0

    global_frame_idx = 0

    print(f"Starting to process {end_index - start_index} labeled videos ...")

    for idx, dr in tqdm(enumerate(labels[start_index:end_index], start=start_index), total=end_index-start_index):
        video_name = dr["data_row"]["external_id"]
        video_url  = dr["data_row"]["row_data"]

        # Collect label/annotation info for each frame from Labelbox
        all_frames_annotations = {}
        for project_id, project_data in dr["projects"].items():
            lb_label = project_data["labels"][0]
            frames_dict = lb_label["annotations"]["frames"]
            for frame_str, frame_data in frames_dict.items():
                all_frames_annotations[int(frame_str)] = frame_data

        # Download the video (NO client.headers => avoid 401)
        try:
            resp = requests.get(video_url)
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"ERROR: Could not download video for {video_name}. HTTPError: {e}")
            continue  # skip entire video

        # Write video to a temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_name = tmp.name

        cap = cv2.VideoCapture(tmp_name)
        local_frame_idx = 0

        while True:
            success, frame_bgr = cap.read()
            if not success:
                break  # no more frames in video
            total_frames_read += 1

            skip_this_frame = False
            # Gather Bolus masks if there's annotation
            bolus_masks = []
            if local_frame_idx in all_frames_annotations:
                frame_data = all_frames_annotations[local_frame_idx]
                for feat_id, feature in frame_data["objects"].items():
                    if feature["name"] == "Bolus":
                        mask_url = feature["mask"]["url"]
                        try:
                            r = requests.get(mask_url, headers=client.headers)
                            r.raise_for_status()
                        except requests.HTTPError as e:
                            print(f"[Frame {local_frame_idx}] Error fetching mask: {e}, skipping frame.")
                            skip_this_frame = True
                            break

                        if not r.content:
                            print(f"[Frame {local_frame_idx}] Empty mask data, skipping frame.")
                            skip_this_frame = True
                            break

                        partial_mask = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
                        if partial_mask is None or partial_mask.size == 0:
                            print(f"[Frame {local_frame_idx}] Could not decode partial mask, skipping.")
                            skip_this_frame = True
                            break

                        bolus_masks.append(partial_mask)

            if skip_this_frame:
                # don't save anything for this frame
                local_frame_idx += 1
                skipped_frames_count += 1
                continue

            # Merge or create black
            h, w = frame_bgr.shape[:2]
            if len(bolus_masks) > 0:
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                merge_failed = False
                for pm in bolus_masks:
                    if pm.shape != (h, w):
                        print(f"[Frame {local_frame_idx}] Mask shape {pm.shape} != frame shape {(h,w)}, skipping frame.")
                        merge_failed = True
                        break
                    try:
                        combined_mask = cv2.add(combined_mask, pm)
                    except cv2.error as e:
                        print(f"[Frame {local_frame_idx}] cv2.add error: {e}, skipping frame.")
                        merge_failed = True
                        break

                if merge_failed:
                    local_frame_idx += 1
                    skipped_frames_count += 1
                    continue
                final_mask = combined_mask
            else:
                # no bolus => black
                final_mask = np.zeros((h, w), dtype=np.uint8)

            # If we reach here => no error => we save the frame + mask
            frame_filename = f"{global_frame_idx}.png"
            mask_filename  = f"{global_frame_idx}_bolus.png"

            cv2.imwrite(os.path.join(frames_output_dir, frame_filename),
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(masks_output_dir, mask_filename),
                        final_mask)

            # Add to data_overview
            data_overview.append({
                "frame_idx": global_frame_idx,
                "video_name": video_name,
                "frame": local_frame_idx
            })

            saved_frames_count += 1
            global_frame_idx += 1
            local_frame_idx  += 1

        cap.release()
        os.remove(tmp_name)
        total_videos += 1

    # End of main loop
    print("\nAll videos processed.")
    print(f"  Total videos:         {total_videos}")
    print(f"  Total frames read:    {total_frames_read}")
    print(f"  Saved frames:         {saved_frames_count}")
    print(f"  Skipped frames:       {skipped_frames_count}")

    # Write final CSV
    if data_overview:
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(data_overview)
        print(f"Data overview CSV written at: {csv_path}")
    else:
        print("No frames saved => no data_overview.csv created.")



def main(api_key, project_id, output_directory):
    """
    Exports labels from a Labelbox project and extracts the single video frames, landmark points, and
    area feature masks from the exported labels by calling the respective helper functions.

    Parameters:
    api_key (str): Labelbox API key to access the Labelbox project
    project_id (str): ID of the project from which the labels should be exported
    output_directory (str): directory where the exported data is saved
    """

    # Initialize the Labelbox client with the provided API key
    client = lb.Client(api_key=api_key)
    # Get the project from the client using the provided project ID
    project = client.get_project(project_id)

    # Recreate the directory '<output_directory>/output' to avoid confusion with the output of previous runs
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    # Create the directories 'frames', 'masks' and 'landmarks' as subdirectories of '<output_directory>/output'
    # for storing the exported video frames and created masks and landmarks
    output_subdirectories = ["imgs", "masks"]
    for sd in output_subdirectories:
        os.makedirs(os.path.join(output_directory, sd))

    # Export the labels from the Labelbox project
    print("\nExporting the labels from the Labelbox project.\n")
    labels = export_labels(project, output_directory)
    export_frames_and_bolus_masks_skip_on_error(
        labels,
        os.path.join(output_directory, "imgs"),
        os.path.join(output_directory, "masks"),
        client,
        os.path.join(output_directory, "data_overview.csv"),
        start_index=0,
        end_index=None)
    '''
    print("\nExporting the labels from the Labelbox project.\n")
    labels = export_labels(project, output_directory)

    # Get a list of the unique annotation categories used in the exported Labelbox labels
    categories = get_annotation_categories(labels)
    print("The exported labels contain the following annotation categories:\n", categories, sep="", end="\n\n")

    # Extract the single video frames, landmark points, and area feature masks from the labels
    print("Extracting the single video frames from the exported labels.\n")
    export_video_frames(labels, os.path.join(output_directory, "frames"))
    #print("Extracting the landmark points from the exported labels.\n")
    #save_landmark_points(labels, os.path.join(output_directory, "landmarks"))
    print("Extracting the area feature masks from the exported labels.\n")
    save_bolus_masks_only(labels, os.path.join(output_directory, "masks"), client)

    print(f"Finished exporting the labels. The output can be found in {output_directory}.\n")
    '''

# The following code block is only executed if this script is being run directly and not imported
# as a module in another script.
# If the code block is run, it parses the command line arguments (Labelbox API key, project ID,
# output directory) and calls the main function with these parsed arguments.
if __name__ == "__main__":
    # Parser for the command line arguments
    # Parser for the command line arguments
    parser = argparse.ArgumentParser(
        description="This scripts exports annotations of data on swallowing events from Labelbox."
    )
    '''
    # Labelbox API key
    parser.add_argument(
        "-ak",
        "--api-key",
        type=str,
        help="API key for accessing Labelbox",
    )
    '''
    # ID of the project from which the data rows should be exported
    # (ID can be found in the URL of the Labelbox project)
    parser.add_argument(
        "-pi",
        "--project-id",
        type=str,
        help="ID of the Labelbox project",
    )
    # Output directory
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        help="Output directory to save exported data",
        default=".",
    )
    # read api key from apikey.txt file
    with open(r"labelbox_export/apikey.txt", "r") as f:
        api_key = f.read().strip()

    # Parse the command line arguments
    args = parser.parse_args()
    api_key = api_key #args.api_key
    project_id = args.project_id
    output_directory = args.output_directory

    # Call the main function with the parsed arguments
    main(api_key, project_id, output_directory)