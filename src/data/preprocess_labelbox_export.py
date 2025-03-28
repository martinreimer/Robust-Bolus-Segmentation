#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to process annotated video exports.

This script:
- Loads data_overview.csv and video_notes.csv.
- Filters video_notes to include only rows where:
    • not_use == 0,
    • the source matches the specified project_source,
    • and bad_quality == 0.
- Prints the number of videos excluded due to the not_use flag.
- Performs an inner join: only videos present in both video_notes (after filtering) and data_overview are processed.
- Prints summary stats on the number of unique videos before processing, after filtering, and the number excluded by the join.
- Processes frames:
    • If a nonzero Desired_last_frame is specified for a video, frames after that are skipped.
    • Copies frames and masks (resizing masks if needed).
    • Accumulates per-video reasons for frame exclusion.
- At the end, prints one summary statement per video that had any excluded frames.
- Writes a new data_overview CSV (excluding the dataset_name column).
"""

import os
import shutil
import pandas as pd
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def process_dataset(
        original_dataset_dir: str,
        video_notes_csv_path: str,
        output_dataset_dir: str,
        project_source: str,
        verbose: bool
):
    # Define input and output directories
    frames_dir = os.path.join(original_dataset_dir, 'imgs')
    masks_dir = os.path.join(original_dataset_dir, 'masks')
    data_overview_csv = os.path.join(original_dataset_dir, 'data_overview.csv')

    new_frames_dir = os.path.join(output_dataset_dir, 'imgs')
    new_masks_dir = os.path.join(output_dataset_dir, 'masks')

    # 1. Load the data_overview CSV
    if not os.path.exists(data_overview_csv):
        raise FileNotFoundError(f"Cannot find data_overview.csv at {data_overview_csv}")
    df_overview = pd.read_csv(data_overview_csv)
    total_frames = len(df_overview)
    original_unique_videos = df_overview['shared_video_id'].unique()
    print(f"Original data_overview: {total_frames} frames from {len(original_unique_videos)} unique videos.")

    # 2. Load and filter the video_notes CSV
    if not os.path.exists(video_notes_csv_path):
        raise FileNotFoundError(f"Cannot find video_notes.csv at {video_notes_csv_path}")
    df_video_notes = pd.read_csv(video_notes_csv_path, sep=";")

    # First, select rows with the desired project_source
    df_video_notes_source = df_video_notes[df_video_notes['source'] == project_source]

    # Identify and report videos excluded due to the not_use flag (for this project_source)
    df_video_notes_excluded = df_video_notes_source[df_video_notes_source['not_use'] != 0]
    excluded_not_use_videos = df_video_notes_excluded['video_id'].unique()
    print(f"Videos excluded due to 'not_use' flag for source {project_source}: {len(excluded_not_use_videos)}")
    if verbose and len(excluded_not_use_videos) > 0:
        for vid in excluded_not_use_videos:
            print(f"  {vid}")

    # Now, keep only videos with not_use == 0 and also exclude bad quality ones.
    df_video_notes_filtered = df_video_notes_source[
        (df_video_notes_source['not_use'] == 0) &
        (df_video_notes_source['bad_quality'] == 0)
    ]
    print(f"After filtering video_notes (not_use==0 and bad_quality==0): {len(df_video_notes_filtered)} rows remain.")

    # Drop duplicates: if there are duplicate entries for a video, keep the first occurrence.
    df_video_notes_unique = df_video_notes_filtered.drop_duplicates(subset='video_id', keep='first')
    print(f"Unique videos in video_notes after filtering: {len(df_video_notes_unique)}")

    # Build a mapping: video_id -> Desired_last_frame (None if missing or 0)
    desired_last_frame_dict = {}
    for _, row in df_video_notes_unique.iterrows():
        video_id = row['video_id']
        try:
            dlf = int(row['Desired_last_frame'])
            desired_last_frame_dict[video_id] = None if dlf == 0 else dlf
        except (ValueError, TypeError):
            desired_last_frame_dict[video_id] = None

    valid_video_ids = set(df_video_notes_unique['video_id'].tolist())

    # 3. Filter the data_overview to only include videos from the filtered video_notes and matching project_source
    df_overview_filtered = df_overview[
        (df_overview['shared_video_id'].isin(valid_video_ids)) &
        (df_overview['project_source'] == project_source)
    ]
    filtered_unique_videos = df_overview_filtered['shared_video_id'].unique()
    print(f"After inner join with video_notes, data_overview has {len(df_overview_filtered)} frames from {len(filtered_unique_videos)} unique videos.")
    excluded_by_inner_join = len(original_unique_videos) - len(filtered_unique_videos)
    print(f"Videos excluded by inner join: {excluded_by_inner_join}")

    # 4. Create new output directories
    os.makedirs(output_dataset_dir, exist_ok=True)
    os.makedirs(new_frames_dir, exist_ok=True)
    os.makedirs(new_masks_dir, exist_ok=True)

    # 5. Process frames with tqdm progress bar and accumulate per-video exclusion info
    new_overview_rows = []
    skipped_frames = 0
    missing_mask_count = 0
    resized_mask_count = 0
    white_mask_count = 0

    # Dictionary to accumulate exclusion reasons per video (one print statement per video later)
    video_exclusion_info = {}

    def add_exclusion(video_id, reason):
        if video_id not in video_exclusion_info:
            video_exclusion_info[video_id] = {}
        if reason not in video_exclusion_info[video_id]:
            video_exclusion_info[video_id][reason] = 0
        video_exclusion_info[video_id][reason] += 1

    for _, row in tqdm(df_overview_filtered.iterrows(), total=len(df_overview_filtered), desc="Processing frames"):
        frame_number = int(row['frame'])     # The frame number in the video sequence
        frame_idx = int(row['frame_idx'])      # Used for the file name (e.g., "0.png")
        video_id = row['shared_video_id']

        # Check if a desired last frame is specified for this video
        desired_last_frame = desired_last_frame_dict.get(video_id, None)
        if desired_last_frame is not None and frame_number > desired_last_frame:
            add_exclusion(video_id, "desired_last_frame")
            skipped_frames += 1
            continue

        # Define file paths for frame and mask
        src_frame_path = os.path.join(frames_dir, f"{frame_idx}.png")
        dst_frame_path = os.path.join(new_frames_dir, f"{frame_idx}.png")

        src_mask_path = os.path.join(masks_dir, f"{frame_idx}_bolus.png")
        dst_mask_path = os.path.join(new_masks_dir, f"{frame_idx}_bolus.png")

        # Copy the frame image
        if os.path.exists(src_frame_path):
            frame_img = cv2.imread(src_frame_path, cv2.IMREAD_UNCHANGED)
            if frame_img is None:
                add_exclusion(video_id, "unreadable_frame")
                skipped_frames += 1
                continue
            cv2.imwrite(dst_frame_path, frame_img)
        else:
            add_exclusion(video_id, "frame_not_found")
            skipped_frames += 1
            continue

        # Process and copy the mask image
        if os.path.exists(src_mask_path):
            mask_img = cv2.imread(src_mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                add_exclusion(video_id, "unreadable_mask")
                missing_mask_count += 1
            else:
                # Skip if the mask is all-white (i.e. every pixel is 255)
                if np.all(mask_img == 255):
                    add_exclusion(video_id, "all_white_mask")
                    white_mask_count += 1
                    if os.path.exists(dst_frame_path):
                        os.remove(dst_frame_path)
                    skipped_frames += 1
                    continue

                # If frame and mask sizes mismatch, resize the mask using nearest-neighbor interpolation
                h_frame, w_frame = frame_img.shape[:2]
                h_mask, w_mask = mask_img.shape[:2]
                if (h_frame, w_frame) != (h_mask, w_mask):
                    mask_img = cv2.resize(mask_img, (w_frame, h_frame), interpolation=cv2.INTER_NEAREST)
                    resized_mask_count += 1

                cv2.imwrite(dst_mask_path, mask_img)
        else:
            add_exclusion(video_id, "missing_mask")
            missing_mask_count += 1

        new_overview_rows.append(row)

    # 6. After processing, print one summary statement per video that had exclusions.
    if video_exclusion_info:
        print("\nFrame exclusion summary per video:")
        for vid, reasons in video_exclusion_info.items():
            reasons_summary = ", ".join([f"{reason}: {count}" for reason, count in reasons.items()])
            print(f"Video {vid} - Excluded frames -> {reasons_summary}")

    # 7. Write the new pruned data_overview CSV without the dataset_name column.
    df_new_overview = pd.DataFrame(new_overview_rows)
    if 'dataset_name' in df_new_overview.columns:
        df_new_overview = df_new_overview.drop(columns=['dataset_name'])
    new_data_overview_path = os.path.join(output_dataset_dir, 'data_overview.csv')
    df_new_overview.to_csv(new_data_overview_path, index=False)

    # 8. Print overall statistics.
    final_frames = len(df_new_overview)
    print("\nProcessing complete!")
    print(f"New dataset created at: {output_dataset_dir}")
    print(f"Total frames in original data_overview: {total_frames}")
    print(f"Total frames retained: {final_frames}")
    print(f"Total frames pruned: {skipped_frames}")
    print(f"Missing or unreadable masks: {missing_mask_count}")
    print(f"Resized mask count: {resized_mask_count}")
    print(f"All-white mask frames removed: {white_mask_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Process annotated video exports and prune frames based on video_notes and data_overview."
    )
    parser.add_argument(
        "--original_dataset_dir",
        required=True,
        help="Path to the original dataset folder (contains imgs, masks, data_overview.csv)."
    )
    parser.add_argument(
        "--video_notes_csv",
        required=True,
        help="Path to the video_notes.csv file."
    )
    parser.add_argument(
        "--output_dataset_dir",
        required=True,
        help="Path where the processed dataset will be saved."
    )
    parser.add_argument(
        "--project_source",
        required=True,
        help="Project source to filter (e.g., MBS or MBSS_Martin)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    args = parser.parse_args()

    process_dataset(
        original_dataset_dir=args.original_dataset_dir,
        video_notes_csv_path=args.video_notes_csv,
        output_dataset_dir=args.output_dataset_dir,
        project_source=args.project_source,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
