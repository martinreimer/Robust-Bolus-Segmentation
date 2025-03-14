"""
This file contains a script to export labels from a Labelbox project and to extract the single video frames,
landmark points, and area feature masks from the exported labels.

Changes:
    - If no mask is found for a frame, we create an all-black mask for that frame.
    - If a bolus mask is corrupt or cannot be fetched or cannot be merged, we write an all-white image (will be removed later)
    - We now read the API key from a file named 'apikey.txt' in the same directory as this script.
    - We now suppress deprecation warnings.
    - The exported CSV now includes additional columns: "id", "dataset_name", "project_source" and "shared_video_id".
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
import argparse
from typing import List, Dict
import csv
from tqdm import tqdm
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

        # Check if any errors occurred during the export
        if task.errors:
            print(f"An error occurred while trying to export the labels: {task.errors}")
            return None

        # Access the result if no errors occurred
        labels = task.result

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

    The resulting CSV has the following columns:
    [frame_idx, video_name, id, dataset_name, project_source, shared_video_id, frame]

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

    # Prepare for CSV logging with additional columns
    csv_header = ["frame_idx", "video_name", "id", "dataset_name", "project_source", "shared_video_id", "frame"]

    # Some counters for summary
    total_videos = 0
    total_frames_read = 0
    saved_frames_count = 0
    skipped_frames_count = 0

    global_frame_idx = 0

    print(f"Starting to process {end_index - start_index} labeled videos ...")

    for idx, dr in tqdm(enumerate(labels[start_index:end_index], start=start_index), total=end_index - start_index):
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

        # Write video to a temporary file for OpenCV
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
            # Gather Bolus masks if there's annotation for this frame
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
                local_frame_idx += 1
                skipped_frames_count += 1
                continue

            # Merge masks or create an all-black mask
            h, w = frame_bgr.shape[:2]
            if len(bolus_masks) > 0:
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                merge_failed = False
                for pm in bolus_masks:
                    if pm.shape != (h, w):
                        print(f"[Frame {local_frame_idx}] Mask shape {pm.shape} != frame shape {(h, w)}, skipping frame.")
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
                # No bolus annotation => create an all-black mask.
                final_mask = np.zeros((h, w), dtype=np.uint8)

            # Save the frame and mask
            frame_filename = f"{global_frame_idx}.png"
            mask_filename  = f"{global_frame_idx}_bolus.png"

            cv2.imwrite(os.path.join(frames_output_dir, frame_filename),
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(masks_output_dir, mask_filename),
                        final_mask)

            # Extract additional data for CSV:
            # Get the unique id and dataset_name from the "data_row" field.
            row_id = dr["data_row"]["id"]
            dataset_name = dr["data_row"].get("details", {}).get("dataset_name", "")
            # Pick the first project's name as the project_source.
            project_source = ""
            if dr.get("projects"):
                project_source = list(dr["projects"].values())[0].get("name", "")

            # Create shared_video_id.
            # For a video name starting with "Example_", remove the prefix and reconstruct the usual video id.
            if video_name.startswith("Example_"):
                # Remove the prefix
                stripped = video_name[len("Example_"):]
                base, ext = os.path.splitext(stripped)
                parts = base.split('_')
                # If there are more than 4 parts, assume the usual video id consists of the first 4 tokens.
                if len(parts) > 4:
                    shared_video_id = '_'.join(parts[:4]) + ext
                else:
                    shared_video_id = stripped
            else:
                shared_video_id = video_name

            # Append row info to CSV data list
            data_row = {
                "frame_idx": global_frame_idx,
                "video_name": video_name,
                "id": row_id,
                "dataset_name": dataset_name,
                "project_source": project_source,
                "shared_video_id": shared_video_id,
                "frame": local_frame_idx
            }
            # Add row to data_overview
            if 'data_overview' not in locals():
                data_overview = []
            data_overview.append(data_row)

            saved_frames_count += 1
            global_frame_idx += 1
            local_frame_idx  += 1

        cap.release()
        os.remove(tmp_name)
        total_videos += 1

    # End of processing loop
    print("\nAll videos processed.")
    print(f"  Total videos:         {total_videos}")
    print(f"  Total frames read:    {total_frames_read}")
    print(f"  Saved frames:         {saved_frames_count}")
    print(f"  Skipped frames:       {skipped_frames_count}")

    # Write final CSV if there are saved frames
    if 'data_overview' in locals() and data_overview:
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

    # Recreate the output directory to avoid confusion with previous runs
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    # Create the directories for storing the exported video frames and masks
    output_subdirectories = ["imgs", "masks"]
    for sd in output_subdirectories:
        os.makedirs(os.path.join(output_directory, sd))

    # Export the labels from the Labelbox project
    print("\nExporting the labels from the Labelbox project.\n")
    labels = export_labels(project, output_directory)
    if labels is None:
        print("No labels were exported. Exiting.")
        return

    export_frames_and_bolus_masks_skip_on_error(
        labels,
        os.path.join(output_directory, "imgs"),
        os.path.join(output_directory, "masks"),
        client,
        os.path.join(output_directory, "data_overview.csv"),
        start_index=0,
        end_index=None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script exports annotations of data on swallowing events from Labelbox."
    )
    # Project ID argument (ID can be found in the URL of the Labelbox project)
    parser.add_argument(
        "-pi",
        "--project-id",
        type=str,
        help="ID of the Labelbox project",
    )
    # Output directory argument
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        help="Output directory to save exported data",
        default="."
    )
    # Read API key from file 'apikey.txt'
    with open(r"labelbox_export/apikey.txt", "r") as f:
        api_key = f.read().strip()

    args = parser.parse_args()
    project_id = args.project_id
    output_directory = args.output_directory

    main(api_key, project_id, output_directory)
