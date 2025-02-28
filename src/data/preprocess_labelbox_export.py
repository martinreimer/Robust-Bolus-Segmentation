#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to prune frames for specified videos based on a curation CSV and then
rebuild a new dataset with copied frames, masks, and a pruned data_overview CSV.
If the mask and frame sizes mismatch, the mask is resized via nearest-neighbor.
Additionally, frames whose masks are all-white (255) are removed.
"""

import os
import shutil
import pandas as pd
import argparse
import cv2
import numpy as np

def prune_and_rebuild_dataset(
        original_dataset_dir: str,
        curation_csv_path: str,
        output_dataset_dir: str,
        verbose: bool
):
    """
    Prune frames for certain videos based on a curation CSV specifying
    `Video_ID` and `Desired_Last_Frame`, then rebuild a new dataset
    (frames, masks, data_overview) into a new folder.
    If a mask doesn't match the frame dimensions, resize it via nearest-neighbor.
    If the mask is all-white (i.e., 255 everywhere), we skip that frame entirely.

    At the end, prints stats on how many frames were skipped, how many masks
    were missing, how many all-white masks were found, etc.
    """

    # Paths
    frames_dir = os.path.join(original_dataset_dir, 'imgs')
    masks_dir = os.path.join(original_dataset_dir, 'masks')
    data_overview_csv = os.path.join(original_dataset_dir, 'data_overview.csv')

    new_frames_dir = os.path.join(output_dataset_dir, 'imgs')
    new_masks_dir = os.path.join(output_dataset_dir, 'masks')

    # 1. Read main data overview CSV
    if not os.path.exists(data_overview_csv):
        raise FileNotFoundError(f"Cannot find data_overview.csv at {data_overview_csv}")
    df_data_overview = pd.read_csv(data_overview_csv)
    total_rows = len(df_data_overview)

    # 2. Read curation CSV (map Video_ID -> Desired_Last_Frame)
    if not os.path.exists(curation_csv_path):
        raise FileNotFoundError(f"Cannot find curation CSV at {curation_csv_path}")
    df_curation = pd.read_csv(curation_csv_path, sep=";")

    if verbose:
        print("Collecting curation info...")

    curation_dict = {}
    for _, row in df_curation.iterrows():
        video_id = str(row['Video_ID'])
        if pd.isna(row['Desired_Last_Frame']):
            continue
        desired_last_frame = int(row['Desired_Last_Frame'])
        curation_dict[video_id] = desired_last_frame
        if verbose:
            print(f"Video {video_id} -> Desired_Last_Frame = {desired_last_frame}")

    # 3. Make new output directories
    os.makedirs(output_dataset_dir, exist_ok=True)
    os.makedirs(new_frames_dir, exist_ok=True)
    os.makedirs(new_masks_dir, exist_ok=True)

    # 4. Build the new data_overview & track stats
    new_overview_rows = []
    skipped_frames = 0
    missing_mask_count = 0
    resized_mask_count = 0
    white_mask_count = 0  # new counter for all-white masks

    for _, row in df_data_overview.iterrows():
        video_frame_idx = int(row['frame'])
        frame_idx = int(row['frame_idx'])
        video_name = row['video_name']

        # Check curation
        if video_name in curation_dict:
            desired_last_frame = curation_dict[video_name]
            if desired_last_frame == 0:
                if verbose:
                    print(f"Skipping all frames for video {video_name}")
                skipped_frames += 1
                continue
            if video_frame_idx > desired_last_frame:
                if verbose:
                    print(f"Skipping {video_name} frame {video_frame_idx} "
                          f"(beyond desired_last_frame={desired_last_frame})")
                skipped_frames += 1
                continue

        # Prepare paths
        src_frame_path = os.path.join(frames_dir, f"{frame_idx}.png")
        dst_frame_path = os.path.join(new_frames_dir, f"{frame_idx}.png")

        src_mask_path = os.path.join(masks_dir, f"{frame_idx}_bolus.png")
        dst_mask_path = os.path.join(new_masks_dir, f"{frame_idx}_bolus.png")

        # Read/copy the frame
        if os.path.exists(src_frame_path):
            frame_img = cv2.imread(src_frame_path, cv2.IMREAD_UNCHANGED)
            if frame_img is None:
                # Could not read
                if verbose:
                    print(f"Warning: could not read frame {src_frame_path}, skipping.")
                skipped_frames += 1
                continue

            # Write frame to new dataset
            cv2.imwrite(dst_frame_path, frame_img)

            # Check the mask
            if os.path.exists(src_mask_path):
                mask_img = cv2.imread(src_mask_path, cv2.IMREAD_UNCHANGED)
                if mask_img is None:
                    # unreadable mask
                    missing_mask_count += 1
                    if verbose:
                        print(f"Warning: could not read mask {src_mask_path}, skipping mask.")
                else:
                    # Check if it's all-white
                    if np.all(mask_img == 255):
                        # skip the entire frame
                        white_mask_count += 1
                        if verbose:
                            print(f"Frame {frame_idx}: all-white mask => removing frame & skipping.")
                        # remove the just-copied frame
                        if os.path.exists(dst_frame_path):
                            os.remove(dst_frame_path)
                        skipped_frames += 1
                        # do NOT copy this row to new_overview_rows
                        continue

                    # If not all-white, check dimension mismatch
                    h_frame, w_frame = frame_img.shape[:2]
                    h_mask, w_mask = mask_img.shape[:2]
                    if (h_frame, w_frame) != (h_mask, w_mask):
                        if verbose:
                            print(f"[Resizing mask] frame {frame_idx}: "
                                  f"frame=({w_frame}x{h_frame}), mask=({w_mask}x{h_mask})")
                        mask_img = cv2.resize(
                            mask_img,
                            (w_frame, h_frame),
                            interpolation=cv2.INTER_NEAREST
                        )
                        resized_mask_count += 1

                    # Finally, write mask to new dataset
                    cv2.imwrite(dst_mask_path, mask_img)
            else:
                # Mask does not exist at all
                missing_mask_count += 1
                if verbose:
                    print(f"No mask found for frame {frame_idx}")

            # Keep this row
            new_overview_rows.append(row)
        else:
            # Frame not found
            if verbose:
                print(f"Frame file not found: {src_frame_path}")
            skipped_frames += 1

    # 5. Write the new data_overview CSV
    df_new_overview = pd.DataFrame(new_overview_rows)
    new_data_overview_path = os.path.join(output_dataset_dir, 'data_overview.csv')
    df_new_overview.to_csv(new_data_overview_path, index=False)

    # Stats
    final_count = len(df_new_overview)
    pruned_count = skipped_frames

    print("\nPruning complete!")
    print(f"New dataset created at: {output_dataset_dir}")
    print(f"Total frames in original data_overview: {total_rows}")
    print(f"Total frames retained: {final_count}")
    print(f"Total frames pruned: {pruned_count}")
    print(f"Missing or unreadable masks: {missing_mask_count}")
    print(f"Resized mask count: {resized_mask_count}")
    print(f"All-white mask count (removed): {white_mask_count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune frames for specified videos based on a curation CSV and rebuild a new dataset."
    )

    parser.add_argument(
        "--original_dataset_dir",
        default=r"D:\Martin\thesis\data\raw\labelbox_output_mbss_martin_0226",
        help="Path to the original dataset folder (containing frames, masks, data_overview.csv)."
    )
    parser.add_argument(
        "--curation_csv",
        default=r"D:\Martin\thesis\data\video_notes.csv",
        help="Path to the CSV that specifies Video_ID and Desired_Last_Frame."
    )
    parser.add_argument(
        "--output_dataset_dir",
        default=r"D:\Martin\thesis\data\processed\labelbox_output_mbss_martin_0226_frames_excluded",
        help="Path where the pruned dataset will be created."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information."
    )

    args = parser.parse_args()

    prune_and_rebuild_dataset(
        original_dataset_dir=args.original_dataset_dir,
        curation_csv_path=args.curation_csv,
        output_dataset_dir=args.output_dataset_dir,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
