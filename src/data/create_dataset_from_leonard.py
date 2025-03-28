
# !/usr/bin/env python3
"""
analyze_mask_frames.py

This script traverses a specified directory, identifies all Flammkuchen-generated .mask HDF5 files,
loads them using Flammkuchen, analyzes the number of frames in each mask, and prints the results.

Usage:
    python analyze_mask_frames.py --directory path/to/mask_files_directory

Example:
    python analyze_mask_frames.py --directory ../../data/raw/video_export/
"""

import os
import argparse
import sys
import flammkuchen as fl
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace  # Importing SimpleNamespace from types module

'''
def find_mask_files(directory, mask_extension=".mask"):
    """
    Traverses the specified directory and finds all files ending with the given mask_extension.

    Parameters:
        directory (str): Path to the directory to search.
        mask_extension (str): File extension to identify mask files.

    Returns:
        list: List of full paths to mask files.
    """
    mask_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(mask_extension.lower()):
                full_path = os.path.join(root, file)
                mask_files.append(full_path)
    return mask_files

def analyze_mask_file(mask_path):
    """
    Loads a mask file using Flammkuchen and determines the number of frames.

    Parameters:
        mask_path (str): Full path to the .mask HDF5 file.

    Returns:
        int: Number of frames in the mask file.
    """
    try:
        data = fl.load(mask_path)

        # Attempt to locate the mask array
        # Adjust the keys based on your actual data structure
        possible_mask_keys = ['mask', 'masks', 'files/mask', 'files/masks']
        mask_array = None

        for key in possible_mask_keys:
            if key in data:
                mask_array = data[key]
                break

        if mask_array is None:
            print(f"[WARNING] No recognizable mask data found in '{mask_path}'. Skipping.")
            return None

        # Check the type and determine number of frames
        if isinstance(mask_array, np.ndarray):
            if mask_array.ndim == 3:
                num_frames = mask_array.shape[0]
                return num_frames
            else:
                print \
                    (f"[WARNING] Mask data in '{mask_path}' does not have 3 dimensions. Expected (frames, height, width). Skipping.")
                return None
        elif isinstance(mask_array, (list, tuple, dict, SimpleNamespace)):
            # If mask_array is a dict or SimpleNamespace, try to find a numpy array inside
            # This part can be customized based on your specific data structure
            if isinstance(mask_array, dict):
                for sub_key, sub_val in mask_array.items():
                    if isinstance(sub_val, np.ndarray):
                        num_frames = sub_val.shape[0]
                        return num_frames
            elif isinstance(mask_array, SimpleNamespace):
                for attr in vars(mask_array):
                    sub_val = getattr(mask_array, attr)
                    if isinstance(sub_val, np.ndarray):
                        num_frames = sub_val.shape[0]
                        return num_frames
        else:
            print(f"[WARNING] Mask data in '{mask_path}' is of unsupported type {type(mask_array)}. Skipping.")
            return None

        print(f"[WARNING] Could not determine number of frames in '{mask_path}'.")
        return None

    except Exception as e:
        print(f"[ERROR] Failed to load '{mask_path}': {e}")
        return None

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Analyze the number of frames in Flammkuchen-generated .mask files.")
    parser.add_argument('--directory', '-d', required=True, help="Path to the directory containing .mask files.")
    parser.add_argument('--mask_extension', '-m', default='.mask', help="File extension for mask files (default: .mask).")
    args = parser.parse_args()

    directory = args.directory
    mask_extension = args.mask_extension

    if not os.path.isdir(directory):
        print(f"[ERROR] The specified directory '{directory}' does not exist or is not a directory.")
        sys.exit(1)

    # Find all mask files in the directory
    print(f"[INFO] Searching for mask files with extension '{mask_extension}' in '{directory}'...")
    mask_files = find_mask_files(directory, mask_extension)

    if not mask_files:
        print(f"[INFO] No mask files with extension '{mask_extension}' found in '{directory}'.")
        sys.exit(0)

    print(f"[INFO] Found {len(mask_files)} mask file(s).")

    # Analyze each mask file
    results = []
    for mask_file in tqdm(mask_files, desc="Analyzing mask files"):
        num_frames = analyze_mask_file(mask_file)
        if num_frames is not None:
            results.append((os.path.basename(mask_file), num_frames))

    # Print the results
    print("\n=== Mask Files Analysis ===")
    if results:
        print(f"{'Mask Filename':<60} | {'Number of Frames'}")
        print("-" * 80)
        fras = 0
        for filename, frames in results:
            print(f"{filename:<60} | {frames}")
            fras += frames
        print(f"Total Frames: {fras}")
    else:
        print("No valid mask data found in any mask files.")

if __name__ == "__main__":
    main()




'''
Not Working
Just use this path: \\fauad.fau.de\shares\ANKI\Datasets\Bolus_normals

import os
import cv2  # pip install opencv-python
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------
# --- USER PARAMETERS ---
# -----------------------

# Path to the folder containing the .mp4 video files and .mp4 mask files
VIDEO_EXPORT_PATH = r"\\fauad.fau.de\shares\ANKI\Projects\Swallowing\Data\from_Leonard\Swallow_Events"




import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import flammkuchen as fl
import sys
import matplotlib.pyplot as plt

def resize_image(img, target_width, target_height, is_mask=False):
    """
    Resizes an image to target_width x target_height.
    - Uses NEAREST interpolation for masks.
    - Uses LANCZOS for regular images.
    """
    if is_mask:
        img = img.resize((target_width, target_height), Image.NEAREST)
    else:
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return img

def extract_frames_from_video(video_path, target_w, target_h, output_imgs_dir, start_id, video_name, data_overview, resolution):
    """
    Extracts frames from a video and saves them as PNG images.
    Returns the updated start_id and updated data_overview DataFrame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return start_id, data_overview

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Extracting {frame_count} frames from {video_path}")

    for frame_idx in tqdm(range(frame_count), desc=f"Extracting frames from {os.path.basename(video_path)}"):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil = resize_image(img_pil, target_w, target_h, is_mask=False)
        frame_filename = f"{start_id}.png"
        img_pil.save(os.path.join(output_imgs_dir, frame_filename))

        # Append to data_overview
        data_overview = data_overview.append({
            "frame_idx": frame_idx,
            "video_name": video_name,
            "frame": frame_idx,  # Alternatively, use timestamp or other info
            "new_frame_name": frame_filename,
            "new_mask_name": f"{start_id}_bolus.png",
            "source": video_path,
            "resolution": resolution
        }, ignore_index=True)

        start_id += 1

    cap.release()
    return start_id, data_overview

def extract_masks_from_flammkuchen(mask_path, target_w, target_h, output_masks_dir, start_id, binary_threshold=1):
    """
    Extracts mask frames from the .mask HDF5 file using Flammkuchen and saves them as PNG images with '_bolus' suffix.
    Returns the updated start_id and the number of masks extracted.
    """
    try:
        # Load the mask data using Flammkuchen
        mask_data = fl.load(mask_path)

        # Attempt to locate the mask array within the loaded data
        # Adjust the key based on your actual data structure
        possible_mask_keys = ['mask', 'masks', 'files/mask', 'files/masks']
        mask_array = None

        for key in possible_mask_keys:
            if key in mask_data:
                mask_array = mask_data[key]
                print(f"[INFO] Found mask data under key '{key}'.")
                break

        if mask_array is None:
            print(f"[WARN] No recognizable mask data found in '{mask_path}'.")
            return start_id, 0

        if not isinstance(mask_array, np.ndarray):
            print(f"[WARN] Mask data under key '{key}' is not a NumPy array. Skipping mask extraction.")
            return start_id, 0

        num_masks = mask_array.shape[0]
        print(f"[INFO] Extracting {num_masks} mask frames from '{mask_path}'.")

        for i in tqdm(range(num_masks), desc=f"Extracting masks from {os.path.basename(mask_path)}"):
            mask_frame = mask_array[i]
            # Binarize the mask
            mask_frame = np.where(mask_frame >= binary_threshold, 255, 0).astype(np.uint8)
            mask_pil = Image.fromarray(mask_frame)
            mask_pil = resize_image(mask_pil, target_w, target_h, is_mask=True)
            mask_filename = f"{start_id}_bolus.png"
            mask_pil.save(os.path.join(output_masks_dir, mask_filename))
            start_id += 1

    except Exception as e:
        print(f"[ERROR] Failed to extract masks from '{mask_path}': {e}")
        return start_id, 0

    return start_id, num_masks

def main():
    parser = argparse.ArgumentParser(description="Create a unified dataset from video exports using Flammkuchen-generated .mask files.")
    parser.add_argument('--video_export_path', '-v', required=True, help="Path to the folder containing .mp4 videos and .mask files.")
    parser.add_argument('--output_dataset_path', '-o', required=True, help="Path to the output unified dataset folder.")
    parser.add_argument('--mask_suffix', '-m', default='', help="Suffix to identify mask files corresponding to videos (default: '_mask').")
    parser.add_argument('--start_id', '-s', type=int, default=0, help="Starting ID for frame naming (default: 0).")
    parser.add_argument('--target_resolution', '-r', default='512x512', help="Target resolution for images (widthxheight, default: '512x512').")
    parser.add_argument('--binary_threshold', '-t', type=int, default=1, help="Threshold to binarize masks (default: 1).")
    args = parser.parse_args()

    video_export_path = args.video_export_path
    output_dataset_path = args.output_dataset_path
    mask_suffix = args.mask_suffix
    frame_id = args.start_id
    target_w, target_h = map(int, args.target_resolution.lower().split('x'))
    binary_threshold = args.binary_threshold

    # Create output directories
    imgs_dir = os.path.join(output_dataset_path, "imgs")
    masks_dir = os.path.join(output_dataset_path, "masks")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Initialize DataFrame
    data_overview = pd.DataFrame(columns=[
        "frame_idx",
        "video_name",
        "frame",
        "new_frame_name",
        "new_mask_name",
        "source",
        "resolution"
    ])

    # List all .mp4 files (excluding mask files)
    all_files = os.listdir(video_export_path)
    main_videos = [f for f in all_files if f.lower().endswith('.mp4') and not f.lower().endswith(f"{mask_suffix}.mp4")]

    print(f"[INFO] Found {len(main_videos)} main videos (excluding mask videos).")

    counter = 0
    for video_file in tqdm(main_videos, desc="[Processing Videos]"):
        counter += 1
        if counter == 10:
            break
        video_path = os.path.join(video_export_path, video_file)
        video_name = os.path.splitext(video_file)[0]

        # Corresponding mask file
        mask_file = f"{video_name}{mask_suffix}.mask"
        mask_path = os.path.join(video_export_path, mask_file)
        has_mask = os.path.isfile(mask_path)

        if has_mask:
            print(f"\n[INFO] Processing video '{video_file}' with corresponding mask '{mask_file}'.")
        else:
            print(f"\n[INFO] Processing video '{video_file}' without a corresponding mask.")

        # Extract frames from video
        # Obtain video resolution from the first frame
        cap_temp = cv2.VideoCapture(video_path)
        if not cap_temp.isOpened():
            print(f"[ERROR] Cannot open video file: {video_path}. Skipping.")
            continue
        ret, frame_bgr = cap_temp.read()
        if not ret:
            print(f"[ERROR] Cannot read frames from video file: {video_path}. Skipping.")
            cap_temp.release()
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        current_resolution = f"{img_pil.width}x{img_pil.height}"
        img_pil.close()
        cap_temp.release()

        # Extract frames
        current_frame_id = frame_id
        frame_id, data_overview = extract_frames_from_video(
            video_path=video_path,
            target_w=target_w,
            target_h=target_h,
            output_imgs_dir=imgs_dir,
            start_id=frame_id,
            video_name=video_name,
            data_overview=data_overview,
            resolution=current_resolution
        )

        # Extract masks if available
        if has_mask:
            frame_id, num_masks = extract_masks_from_flammkuchen(
                mask_path=mask_path,
                target_w=target_w,
                target_h=target_h,
                output_masks_dir=masks_dir,
                start_id=frame_id,
                binary_threshold=binary_threshold
            )

            # Verify that number of masks matches number of frames
            cap_video = cv2.VideoCapture(video_path)
            if not cap_video.isOpened():
                print(f"[ERROR] Cannot open video file: {video_path}. Skipping mask verification.")
                continue
            num_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_video.release()
            if num_masks != num_frames:
                print(f"[WARN] Number of masks ({num_masks}) does not match number of frames ({num_frames}) in '{video_file}'.")
        else:
            # Create black masks for each frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video file: {video_path}. Skipping mask creation.")
                continue

            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[INFO] Creating black masks for {num_frames} frames in '{video_file}'.")

            for frame_idx in tqdm(range(num_frames), desc=f"Creating black masks for {video_file}"):
                mask_pil = Image.new("L", (target_w, target_h), color=0)
                mask_filename = f"{frame_id}_bolus.png"
                mask_pil.save(os.path.join(masks_dir, mask_filename))

                # Append to data_overview
                data_overview = data_overview.append({
                    "frame_idx": frame_idx,
                    "video_name": video_name,
                    "frame": frame_idx,  # Alternatively, use timestamp or other info
                    "new_frame_name": f"{frame_id}.png",
                    "new_mask_name": f"{frame_id}_bolus.png",
                    "source": video_path,
                    "resolution": current_resolution
                }, ignore_index=True)

                frame_id += 1

            cap.release()

    # Save data_overview.csv
    overview_csv_path = os.path.join(output_dataset_path, "data_overview.csv")
    data_overview.to_csv(overview_csv_path, index=False)
    print(f"\n[INFO] Created unified dataset with frames up to ID {frame_id - 1}.")
    print(f"[INFO] data_overview.csv saved at '{overview_csv_path}'.")

    print("\n[INFO] Done creating unified dataset!")

if __name__ == "__main__":
    main()

'''