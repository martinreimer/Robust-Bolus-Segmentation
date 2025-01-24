"""
python create_train_test_split.py --input_dirs ../../data/processed/leonard_dataset ../../data/processed/dataset_labelbox_export_0124 --output_dir ../../data/processed/train_test_split

This script processes image segmentation datasets, splitting them into train and test sets.
It handles datasets that may or may not include metadata in a `data_overview.csv` file.

Input Assumptions:
- Each input dataset directory contains:
  - An `imgs/` folder with grayscale image frames (e.g., `0.png`, `1.png`).
  - A `masks/` folder with corresponding segmentation masks (e.g., `0_bolus.png`).
  - An optional `data_overview.csv` file with metadata for frames.
    - Required columns (if available):
      - `video_name`: Name of the video the frame belongs to.
      - `frame_idx`: Original frame number within the video.
      - `new_frame_name`: Original frame filename (e.g., `0.png`).
      - `new_mask_name`: Original mask filename (e.g., `0_bolus.png`).

Functionality:
- If `data_overview.csv` is found:
  - Splits frames based on unique `video_name`, ensuring all frames of a video are in the same split.
  - Creates a new `data_overview.csv` in the output directory with tracking information.
- If `data_overview.csv` is **not** found:
  - Assumes frames are sequentially ordered.
  - Performs a sequential split without randomness to keep consecutive frames together.

Output:
- The script organizes the data into the specified output directory with the structure:
  - `train/imgs/`, `train/masks/`
  - `test/imgs/`, `test/masks/`
  - `data_overview.csv` with:
    - `video_name`, `frame_idx`, `old_frame_name`, `new_frame_name`, `split`, `source_dataset`

Usage Example:
    python create_train_test_split_flexible.py \
        --input_dirs path/to/dataset1 path/to/dataset2 \
        --output_dir path/to/output_split \
        --test_size 0.2 \
        --random_seed 42
"""

import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
import math
import random


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create train/test splits for datasets with or without data_overview.csv.")
    parser.add_argument('--input_dirs', '-i', nargs='+', required=True, help="Paths to input dataset directories.")
    parser.add_argument('--output_dir', '-o', required=True, help="Output directory for train/test splits.")
    parser.add_argument('--test_size', '-t', type=float, default=0.2,
                        help="Fraction to allocate to the test set (default: 0.2).")
    parser.add_argument('--random_seed', '-s', type=int, default=42, help="Seed for random operations (default: 42).")
    return parser.parse_args()


def create_directory_structure(output_dir):
    """Creates the necessary directory structure for train and test splits."""
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)


def copy_files(file_list, src_dir, dest_imgs_dir, dest_masks_dir, split, dataset_name, data_overview):
    """Copies image and mask files and updates the overview log."""
    for img, mask, old_frame_name, video_name, frame_idx in tqdm(file_list, desc=f"Copying {split} data"):
        new_img_name = f"{len(data_overview)}.png"
        new_mask_name = f"{len(data_overview)}_bolus.png"

        shutil.copy2(os.path.join(src_dir, "imgs", img), os.path.join(dest_imgs_dir, new_img_name))
        shutil.copy2(os.path.join(src_dir, "masks", mask), os.path.join(dest_masks_dir, new_mask_name))

        # Update data overview
        data_overview.append({
            "video_name": video_name if video_name else '',
            "frame_idx": frame_idx if frame_idx else '',
            "old_frame_name": old_frame_name,
            "new_frame_name": new_img_name,
            "split": split,
            "source_dataset": dataset_name
        })


def split_dataset_with_overview(data_overview, test_size, random_seed, dataset_name):
    """Splits the dataset by video names if overview file is present."""
    grouped = data_overview.groupby('video_name')
    videos = [(name, group) for name, group in grouped]

    random.seed(random_seed)
    random.shuffle(videos)

    test_videos_count = math.ceil(len(videos) * test_size)
    test_videos = videos[:test_videos_count]
    train_videos = videos[test_videos_count:]

    train_files, test_files = [], []

    for _, group in train_videos:
        for _, row in group.iterrows():
            train_files.append((row['new_frame_name'], row['new_mask_name'], row['new_frame_name'], row['video_name'],
                                row['frame_idx']))

    for _, group in test_videos:
        for _, row in group.iterrows():
            test_files.append((row['new_frame_name'], row['new_mask_name'], row['new_frame_name'], row['video_name'],
                               row['frame_idx']))

    return train_files, test_files


def split_dataset_without_overview(src_dir, test_size):
    """Splits the dataset sequentially if no overview file is present."""
    imgs = sorted([f for f in os.listdir(os.path.join(src_dir, "imgs")) if f.endswith('.png') and not f.endswith('_bolus.png')])
    masks = sorted([f for f in os.listdir(os.path.join(src_dir, "masks")) if f.endswith('_bolus.png')])

    if len(imgs) != len(masks):
        print("[WARNING] Number of images and masks do not match. Skipping unmatched items.")
        min_len = min(len(imgs), len(masks))
        imgs = imgs[:min_len]
        masks = masks[:min_len]

    total_frames = len(imgs)
    test_frames_count = math.ceil(total_frames * test_size)

    train_frames = [(imgs[i], masks[i], imgs[i], None, None) for i in range(total_frames - test_frames_count)]
    test_frames = [(imgs[i], masks[i], imgs[i], None, None) for i in
                   range(total_frames - test_frames_count, total_frames)]

    return train_frames, test_frames


def main():
    args = parse_arguments()
    create_directory_structure(args.output_dir)
    data_overview = []

    for input_dir in args.input_dirs:
        print(f"\nProcessing dataset: {input_dir}")
        dataset_name = os.path.basename(os.path.normpath(input_dir))

        overview_path = os.path.join(input_dir, 'data_overview.csv')
        if os.path.isfile(overview_path):
            print("[INFO] Found data_overview.csv. Splitting by video names.")
            try:
                df_overview = pd.read_csv(overview_path)
            except Exception as e:
                print(f"[ERROR] Could not read {overview_path}: {e}. Skipping dataset.")
                continue

            if {'frame_idx', 'video_name', 'new_frame_name', 'new_mask_name'}.issubset(df_overview.columns):
                train_files, test_files = split_dataset_with_overview(df_overview, args.test_size, args.random_seed,
                                                                      dataset_name)
            else:
                print(f"[ERROR] Missing required columns in {overview_path}. Skipping dataset.")
                continue
        else:
            print("[INFO] No data_overview.csv found. Splitting sequentially by frames.")
            train_files, test_files = split_dataset_without_overview(input_dir, args.test_size)

        # Copy train files
        copy_files(train_files, input_dir, os.path.join(args.output_dir, 'train', 'imgs'),
                   os.path.join(args.output_dir, 'train', 'masks'), 'train', dataset_name, data_overview)

        # Copy test files
        copy_files(test_files, input_dir, os.path.join(args.output_dir, 'test', 'imgs'),
                   os.path.join(args.output_dir, 'test', 'masks'), 'test', dataset_name, data_overview)

    # Save the consolidated data overview
    df_final_overview = pd.DataFrame(data_overview)
    df_final_overview.to_csv(os.path.join(args.output_dir, 'data_overview.csv'), index=False)

    print("\n[INFO] Train-test split completed successfully.")
    print(f"[INFO] Overview saved to: {os.path.join(args.output_dir, 'data_overview.csv')}")


if __name__ == "__main__":
    main()
