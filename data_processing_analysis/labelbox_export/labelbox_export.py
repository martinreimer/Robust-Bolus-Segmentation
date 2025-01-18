"""
This file contains a script to export labels from a Labelbox project and to extract the single video frames,
landmark points, and area feature masks from the exported labels.
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
import tempfile
import requests
import argparse
from typing import List, Dict
import csv


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


def get_annotation_categories(labels):
    """
    Creates a list of unique annotation categories used in the provided Labelbox labels.

    Parameters:
    - labels (list): list of Labelbox labels (return value of the function 'export_labels')

    Returns:
    - list: list of unique annotation categories used in 'labels'
    """

    # Create an empty set to store unique category names
    unique_categories = set()

    # Iterate through the data to find the category names
    for dr in labels:
        for project_id, project_data in dr["projects"].items():
            for label in project_data["labels"]:
                for frame_data in label["annotations"]["frames"].values():
                    for feature in frame_data["objects"].values():
                        unique_categories.add(feature["name"])

    # Convert the set to a list for easier access
    unique_categories_list = list(unique_categories)

    return unique_categories_list


def export_video_frames(labels, output_directory, start_index=0, end_index=None) -> None:
    """
    Exports the single video frames from the provided Labelbox labels as .png images and
    creates 'data_overview.csv' as overview of the exported frames.

    Parameters:
    - labels (list): list of Labelbox labels (return value of the function 'export_labels')
    - output_directory (str): directory where the exported frames and the .csv file are saved
    - start_index: index at which to start processing the list of Labelbox labels (default: 0)
    - end_index: index at which to stop processing the list of Labelbox labels (default: len('labels')))
    """
    # Header for the .csv file
    csv_header = ["frame_idx", "video_name", "frame"]
    # List to store the data for the .csv file
    data_overview = []
    # Counter for the total number of exported frames
    global_frame_idx = 0

    # Set end_index to the length of the label list if no other value is provided
    if end_index is None:
        end_index = len(labels)

    # Iterate over the data
    for dr in labels[start_index:end_index]:
        data = dr["data_row"]
        video_name = data["external_id"]
        video_url = data["row_data"]

        # Download the video belonging to the current label and store it in a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            response = requests.get(video_url)
            response.raise_for_status()
            temp_file.write(response.content)

            # Open the temporary video file
            video_capture = cv2.VideoCapture(temp_file.name)

            # Counter for the number of frames exported from the current video
            frame_idx = 0
            while True:
                # Read the next frame from the video
                success, frame = video_capture.read()
                # Exit the loops if no more frames are available
                if not success:
                    break

                # Save the current frame as '<global_frame_idx>.png'
                image_filename = f"{global_frame_idx}.png"
                image_path = os.path.join(output_directory, image_filename)
                cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Append the global frame index, the name of the current video and the index
                # of the frame with regard to the current video to 'data_overview'
                data_overview.append(
                    {"frame_idx": global_frame_idx, "video_name": video_name, "frame": frame_idx}
                )

                frame_idx += 1
                global_frame_idx += 1

        # Explicitly release the video capture object
        video_capture.release()

        # Write the information stored in 'data_overview' to 'data_overview.csv'
        with open(
                f"{output_directory}/../data_overview.csv", mode="w", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(data_overview)


def save_landmark_points(labels, output_directory, start_index=0, end_index=None) -> None:
    """
    Extracts key anatomical points from the provided Labelbox labels and saves them as .png images.

    Parameters:
    - labels (list): list of Labelbox labels (return value of the function 'export_labels')
    - output_directory (str): directory where the landmarks are saved
    - start_index: index at which to start processing the list of Labelbox labels (default: 0)
    - end_index: index at which to stop processing the list of Labelbox labels (default: len('labels')))
    """

    # Set end_index to the length of the label list if no other value is provided
    if end_index is None:
        end_index = len(labels)

    # List containing the names of the different anatomical points
    landmark_list = ["BaseOfVallecula", "Epiglottis", "NasalSpine", "Arytenoids", "Cricoid", "Hyoid"]

    # Counter for the total number of processed frames
    global_frame_idx = 0

    # Iterate over the data
    for dr in labels[start_index:end_index]:
        video_name = dr["data_row"]["external_id"]
        # Iterate over each project that the current data row is associated with
        for project_id, project_data in dr["projects"].items():
            # Get the label of the current data row in the current project
            label = project_data["labels"][0]
            # Iterate over the annotations of each frame of the current label
            for frame, frame_annotations in label["annotations"]["frames"].items():
                # Iterate over the features of the annotation of the current frame
                for feature_id, feature in frame_annotations["objects"].items():
                    # If the current feature is a landmark point
                    if feature["name"] in landmark_list:
                        # Extract the x and y coordinates of the point
                        x_coord = feature["point"]["x"]
                        y_coord = feature["point"]["y"]

                        # Create a black image with the same width and height as the video frames
                        width, height = (
                            dr["media_attributes"]["width"],
                            dr["media_attributes"]["height"],
                        )
                        image = np.zeros((height, width), dtype=np.uint8)

                        # Mark the position of the landmark point with a white circle on the black image
                        x, y = int(x_coord), int(y_coord)
                        result_image = cv2.circle(image, (x, y), 5, (255), -1)

                        # Save the image as '<global_frame_idx>_<f_name>.png' where 'f_name' is the
                        # name of the current anatomical point
                        f_name = feature["name"]
                        image_filename = f"{global_frame_idx}_{f_name.lower()}.png"
                        image_path = os.path.join(output_directory, image_filename)
                        cv2.imwrite(image_path, result_image)

                global_frame_idx += 1


def save_masks(labels: List[Dict], output_directory: str, client, start_index=0, end_index=None):
    """
    Exports masks of the bolus from Labelbox and creates masks of the other area features for each
    video frame in the provided Labelbox labels. All masks are saved as .png images.

    Parameters:
    - labels (list): list of Labelbox labels (return value of the function 'export_labels')
    - output_directory (str): directory where the masks are saved
    - client (labelbox.client.Client): Labelbox client to export the Bolus masks from Labelbox
    - start_index (int): index at which to start processing the list of Labelbox labels (default: 0)
    - end_index (int): index at which to stop processing the list of Labelbox labels (default: len('labels')))
    """

    # Set end_index to the length of the label list if no other value is provided
    if end_index is None:
        end_index = len(labels)

    # List containing the names of the different area features
    area_features_list = ["Pharynx", "SpineArea", "Esophagus", "Airway", "OralArea"]

    # Counter for the total number of processed frames
    global_frame_idx = 0

    # Iterate over the data
    for dr in labels[start_index:end_index]:
        video_name = dr["data_row"]["external_id"]
        # Iterate over each project that the current data row is associated with
        for project_id, project_data in dr["projects"].items():
            # Get the label of the current data row in the current project
            label = project_data["labels"][0]
            # Iterate over the annotations of each frame of the current label
            for frame, frame_data in label["annotations"]["frames"].items():
                # Counter for the subfeatures (single vertebrae) of the 'SpineArea' feature
                spinearea_counter = 1

                # List for collecting the bolus masks of the current frame
                # (there could be more than one bolus mask for a single frame)
                bolus_masks = []

                # Iterate over the features of the annotation of the current frame
                features_list = list(frame_data["objects"].values())
                for feature_idx in range(len(features_list)):
                    feature = features_list[feature_idx]

                    # If the current feature is a bolus mask
                    if feature["name"] == "Bolus":

                        # Fetch the bolus mask from the Labelbox website
                        mask_url = feature["mask"]["url"]
                        mask_data = requests.get(mask_url, headers=client.headers).content
                        if mask_data:
                            img_stream = BytesIO(mask_data)
                            mask = cv2.imdecode(
                                np.frombuffer(img_stream.read(), np.uint8),
                                cv2.IMREAD_GRAYSCALE,
                            )
                            # Append the mask to the list of bolus masks of the current frame
                            bolus_masks.append(mask)

                    # If the current feature is another area feature
                    if feature["name"] in area_features_list:
                        # Extract the coordinates of the points defining the feature's outline
                        pts = [(point["x"], point["y"]) for point in feature["line"]]
                        pts.append(pts[0])
                        pts = np.array(pts, dtype=np.int32)

                        # Create a black image with the same width and height as the video frames
                        width, height = dr["media_attributes"]["width"], dr["media_attributes"]["height"]
                        mask = np.zeros((height, width), dtype=np.uint8)

                        # Fill the inside of the feature's outline with white pixels to create a mask
                        mask = cv2.fillPoly(mask, [pts], 255)

                        # Save the created masks as .png image
                        if feature["name"] == "SpineArea":
                            image_filename = f"{global_frame_idx}_spinearea_{spinearea_counter}.png"
                            spinearea_counter += 1
                        else:
                            f_name = feature["name"]
                            image_filename = f"{global_frame_idx}_{f_name.lower()}.png"
                        image_path = os.path.join(output_directory, image_filename)
                        cv2.imwrite(image_path, mask)

                # If there are any bolus masks for the current frame
                if len(bolus_masks) > 0:
                    # Create a black image with the same width and height as the video frames for
                    # merging the (potentially multiple) bolus masks of the current frame
                    mask = np.zeros_like(bolus_masks[0], dtype=np.uint8)

                    # Add all of the bolus masks to the created black image
                    merging_successful = True
                    for partial_mask in bolus_masks:
                        try:
                            mask = cv2.add(mask, partial_mask)
                        except cv2.error as error:
                            print(f"Error combining bolus masks in frame {global_frame_idx}: {error}")
                            # If the merging causes an error, skip this iteration and continue with the next one
                            merging_successful = False
                            continue

                            # If the merging was successful, save the merged bolus mask as .png image
                    if merging_successful:
                        image_filename = f"{global_frame_idx}_bolus.png"
                        cv2.imwrite(os.path.join(output_directory, image_filename), mask)
                    # Else retry fetching the bolus masks
                    else:
                        feature_idx -= 1

                global_frame_idx += 1


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
    output_directory = os.path.join(output_directory, "output")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    # Create the directories 'frames', 'masks' and 'landmarks' as subdirectories of '<output_directory>/output'
    # for storing the exported video frames and created masks and landmarks
    output_subdirectories = ["frames", "masks", "landmarks"]
    for sd in output_subdirectories:
        os.makedirs(os.path.join(output_directory, sd))

    # Export the labels from the Labelbox project
    print("\nExporting the labels from the Labelbox project.\n")
    labels = export_labels(project, output_directory)

    # Get a list of the unique annotation categories used in the exported Labelbox labels
    categories = get_annotation_categories(labels)
    print("The exported labels contain the following annotation categories:\n", categories, sep="", end="\n\n")

    # Extract the single video frames, landmark points, and area feature masks from the labels
    print("Extracting the single video frames from the exported labels.\n")
    export_video_frames(labels, os.path.join(output_directory, "frames"))
    print("Extracting the landmark points from the exported labels.\n")
    save_landmark_points(labels, os.path.join(output_directory, "landmarks"))
    print("Extracting the area feature masks from the exported labels.\n")
    save_masks(labels, os.path.join(output_directory, "masks"), client)

    print("Finished exporting the labels. The output can be found in './output'.\n")


# The following code block is only executed if this script is being run directly and not imported
# as a module in another script.
# If the code block is run, it parses the command line arguments (Labelbox API key, project ID,
# output directory) and calls the main function with these parsed arguments.
if __name__ == "__main__":
    # Parser for the command line arguments
    parser = argparse.ArgumentParser(
        description="This scripts exports annotations of data on swallowing events from Labelbox."
    )
    # Labelbox API key
    parser.add_argument(
        "-ak",
        "--api-key",
        type=str,
        help="API key for accessing Labelbox",
    )
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

    # Parse the command line arguments
    args = parser.parse_args()
    api_key = args.api_key
    project_id = args.project_id
    output_directory = args.output_directory

    # Call the main function with the parsed arguments
    main(api_key, project_id, output_directory)