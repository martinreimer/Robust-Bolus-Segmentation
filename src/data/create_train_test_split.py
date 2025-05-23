"""
python create_train_test_split.py --input_dirs ../../data/processed/leonard_dataset ../../data/processed/dataset_labelbox_export_0124 --output_dir ../../data/processed/dataset_train_val_test_split

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
from PIL import Image

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for datasets with or without data_overview.csv.")
    parser.add_argument('--input_dirs', '-i', nargs='+', required=True, help="Paths to input dataset directories.")
    parser.add_argument('--output_dir', '-o', required=True, help="Output directory for train/val/test splits.")
    parser.add_argument('--test_size', '-t', type=float, default=0.15, help="Fraction for the test set (default: 0.2).")
    parser.add_argument('--val_size', '-v', type=float, default=0.15,
                        help="Fraction for the validation set (default: 0.1).")
    parser.add_argument('--random_seed', '-s', type=int, default=42, help="Seed for random operations (default: 42).")
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Creates the necessary directory structure for train, val, and test splits."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)


def copy_files(file_list, src_dir, dest_imgs_dir, dest_masks_dir, split, dataset_name, data_overview):
    """Copies image and mask files, resizes them, and updates the overview log."""
    for patient_id, frame_id, video_name, project_source in tqdm(file_list, desc=f"Processing {split} data"):
        #img, mask, old_frame_name, video_name, frame_idx
        img_name = f"{frame_id}.png"
        mask_name = f"{frame_id}_bolus.png"

        old_img_path = os.path.join(src_dir, "imgs", img_name)
        old_mask_path = os.path.join(src_dir, "masks", mask_name)

        new_img_path = os.path.join(dest_imgs_dir, img_name)
        new_mask_path = os.path.join(dest_masks_dir, mask_name)

        shutil.copy(old_img_path, new_img_path)
        # Copy or create a black mask if mask doesn't exist

        if os.path.exists(old_mask_path):
            shutil.copy(old_mask_path, new_mask_path)
        else:
            img = Image.open(new_img_path)
            w, h = img.size
            blank_mask = Image.new("L", (w, h), color=0)
            blank_mask.save(new_mask_path)



        data_overview.append({
            "patient_id": patient_id,
            "video_name": video_name,
            "frame_idx": frame_id,
            "split": split,
            "source_dataset": dataset_name,
        })

def report_videos_per_patient(df) -> None:
    """
    Reads a CSV with columns including 'patient_id' and 'video_name',
    then prints, for each patient_id, the count of unique video_names
    and the list of those names, sorted by count descending.
    """
    # Group by patient_id and collect unique video_names
    print("[INFO] Reporting videos per patient...")
    grouped = (
        df.groupby('patient_id')['video_name']
        .agg(lambda names: sorted(set(names)))
        .reset_index(name='videos')
    )
    # Count how many unique videos per patient
    grouped['video_count'] = grouped['videos'].str.len()
    # Sort descending by count
    grouped = grouped.sort_values('video_count', ascending=False)

    # Print per-patient details
    for _, row in grouped.iterrows():
        vids = ', '.join(row['videos'])
        print(f"Patient {row['patient_id']} ({row['video_count']} videos): {vids}")

    # Build and print distribution of video_count
    dist = grouped['video_count'].value_counts().sort_index()
    print("\nDistribution of video counts:")
    for num_videos, num_patients in dist.items():
        print(
            f"  {num_videos} video{'s' if num_videos != 1 else ''}: {num_patients} patient{'s' if num_patients != 1 else ''}")
    print("\n\n")


def split_dataset_with_overview(data_overview, test_size, val_size, random_seed, dataset_name):
    """Splits the dataset by video names if overview file is present."""
    report_videos_per_patient(df=data_overview)
    # print out how many videos we have per patient id

    grouped = data_overview.groupby('patient_id')
    videos = [(name, group) for name, group in grouped]
    print(f"[INFO] Found {len(videos)} unique Patient IDs in the dataset. In total we have {len(data_overview)} frames.")
    random.seed(random_seed)
    random.shuffle(videos)

    test_videos_count = math.ceil(len(videos) * test_size)
    val_videos_count = math.ceil(len(videos) * val_size)

    test_videos = videos[:test_videos_count]
    val_videos = videos[test_videos_count:test_videos_count + val_videos_count]
    train_videos = videos[test_videos_count + val_videos_count:]

    train_files, val_files, test_files = [], [], []

    # Create a mapping of video names to their respective dataframes
    num_patient_ids, num_videos, num_frames = 0, 0, 0
    for _, group in train_videos:
        num_videos += len(list(group['video_name'].unique()))
        num_frames += len(group)
        num_patient_ids += 1
        for _, row in group.iterrows():
            train_files.append((row['patient_id'], row['frame_idx'], row['video_name'], row['project_source']))
    print(f"[INFO] Train Split contains {num_patient_ids} patients with {num_videos} unique videos with {num_frames} frames.")

    num_patient_ids, num_videos, num_frames = 0, 0, 0
    for _, group in val_videos:
        num_videos += len(list(group['video_name'].unique()))
        num_frames += len(group)
        num_patient_ids += 1
        for _, row in group.iterrows():
            val_files.append((row['patient_id'], row['frame_idx'], row['video_name'], row['project_source']))
    print(f"[INFO] Val Split contains {num_patient_ids} patients with {num_videos} unique videos with {num_frames} frames.")

    num_patient_ids, num_videos, num_frames = 0, 0, 0
    for _, group in test_videos:
        num_videos += len(list(group['video_name'].unique()))
        num_frames += len(group)
        num_patient_ids += 1
        for _, row in group.iterrows():
            test_files.append((row['patient_id'], row['frame_idx'], row['video_name'], row['project_source']))
    print(f"[INFO] Test Split contains {num_patient_ids} patients with {num_videos} unique videos with {num_frames} frames.")

    return train_files, val_files, test_files


def split_dataset_without_overview(src_dir, test_size, val_size):
    """Splits the dataset sequentially if no overview file is present."""
    imgs = sorted(
        [f for f in os.listdir(os.path.join(src_dir, "imgs")) if f.endswith('.png') and not f.endswith('_bolus.png')])
    masks = sorted([f for f in os.listdir(os.path.join(src_dir, "masks")) if f.endswith('_bolus.png')])

    total_frames = len(imgs)
    test_count = math.ceil(total_frames * test_size)
    val_count = math.ceil(total_frames * val_size)

    train_frames = [(imgs[i], masks[i], imgs[i], None, None) for i in range(total_frames - test_count - val_count)]
    val_frames = [(imgs[i], masks[i], imgs[i], None, None) for i in
                  range(total_frames - test_count - val_count, total_frames - test_count)]
    test_frames = [(imgs[i], masks[i], imgs[i], None, None) for i in range(total_frames - test_count, total_frames)]

    return train_frames, val_frames, test_frames


def main():
    args = parse_arguments()
    create_directory_structure(args.output_dir)
    data_overview = []

    for input_dir in args.input_dirs:
        print(f"\nProcessing dataset: {input_dir}")
        dataset_name = os.path.basename(os.path.normpath(input_dir))

        overview_path = os.path.join(input_dir, 'data_overview.csv')
        if os.path.isfile(overview_path):
            df_overview = pd.read_csv(overview_path)
            train_files, val_files, test_files = split_dataset_with_overview(df_overview, args.test_size, args.val_size, args.random_seed, dataset_name)
        else:
            train_files, val_files, test_files = split_dataset_without_overview(input_dir, args.test_size,  args.val_size)

        copy_files(train_files, input_dir, os.path.join(args.output_dir, 'train', 'imgs'),
                   os.path.join(args.output_dir, 'train', 'masks'), 'train', dataset_name, data_overview)
        copy_files(val_files, input_dir, os.path.join(args.output_dir, 'val', 'imgs'),
                   os.path.join(args.output_dir, 'val', 'masks'), 'val', dataset_name, data_overview)
        copy_files(test_files, input_dir, os.path.join(args.output_dir, 'test', 'imgs'),
                   os.path.join(args.output_dir, 'test', 'masks'), 'test', dataset_name, data_overview)

    df_final_overview = pd.DataFrame(data_overview)
    df_final_overview.to_csv(os.path.join(args.output_dir, 'data_overview.csv'), index=False)

    print("\n[INFO] Train-val-test split completed successfully.")


if __name__ == "__main__":
    main()
