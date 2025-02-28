#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline script that replicates the series of steps from the provided shell script.
It runs each command in sequence, checks for shape mismatches right after each
Labelbox export and pruning, and then continues the subsequent processing.

Default arguments match your shell script. You can override them on the command line.

Example:
--------
python pipeline.py
    (runs everything with default paths)

python pipeline.py --skip_export
    (skips the labelbox export fetching + raw shape checks but runs pruning + all else)

python pipeline.py --path_prefix "D:/OtherLocation" --final_dataset_name "dataset_0301"
    (overrides path prefix and final dataset name)
"""
import argparse
import subprocess
import sys
import os

def run_command(cmd: str):
    """
    Helper function to run a command-line string.
    Raises an exception if the command fails (subprocess.run with check=True).
    """
    print(f"\n--- RUNNING COMMAND ---\n{cmd}\n")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="ETL Pipeline for Labelbox data, with shape mismatch checks."
    )

    # Base paths / configuration
    parser.add_argument(
        "--path_prefix",
        default="D:/Martin/thesis/data",
        help="Base path prefix for raw and processed data."
    )
    parser.add_argument(
        "--final_dataset_name",
        default="dataset_0227",
        help="Name for the merged/final dataset folder (under 'processed')."
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="If set, skip the Labelbox export fetching steps (and related shape checks)."
    )

    args = parser.parse_args()

    # Construct default paths
    PATH_PREFIX = args.path_prefix.rstrip("/").rstrip("\\")

    RAW_DIR_MBSS = f"{PATH_PREFIX}/raw/labelbox_output_mbss_martin"
    RAW_DIR_MBS  = f"{PATH_PREFIX}/raw/labelbox_output_mbs"

    PRUNED_MBSS = f"{PATH_PREFIX}/processed/labelbox_output_mbss_martin_frames_excluded"
    PRUNED_MBS  = f"{PATH_PREFIX}/processed/labelbox_output_mbs_frames_excluded"

    # For the final merged dataset
    FINAL_DATASET_NAME = args.final_dataset_name
    FINAL_DATASET_PATH = f"{PATH_PREFIX}/processed/{FINAL_DATASET_NAME}"

    # Let's define the curation CSV path (for pruning)
    CURATION_CSV = f"{PATH_PREFIX}/video_notes.csv"

    # 1) Export from Labelbox (Project: mbss_martin)
    if not args.skip_export:
        print("=== 1. Export from Labelbox (Project: mbss_martin) ===")
        cmd_export_mbss = (
            f"python labelbox_export/main.py "
            f"-pi cm5qm2z0c0b8b07zgdc5f52jp "
            f"-o \"{RAW_DIR_MBSS}\""
        )
        run_command(cmd_export_mbss)

        # Check shape mismatches (mbss_martin RAW)
        print("=== Check shape mismatches: mbss_martin RAW export ===")
        cmd_check_mbss_raw = (
            f"python check_shape_mismatches.py "
            f"--path \"{RAW_DIR_MBSS}\" "
            f"--img_folder \"imgs\" "
            f"--mask_folder \"masks\" "
            f"--mask_suffix \"_bolus\""
        )
        run_command(cmd_check_mbss_raw)
    else:
        print("Skipping Labelbox export for mbss_martin (and RAW mismatch checks).")

    # 2) Export from Labelbox (Project: mbs)
    if not args.skip_export:
        print("=== 2. Export from Labelbox (Project: mbs) ===")
        cmd_export_mbs = (
            f"python labelbox_export/main.py "
            f"-pi cl2t3p57f1b5y0764dutw9c2t "
            f"-o \"{RAW_DIR_MBS}\""
        )
        run_command(cmd_export_mbs)

        # Check shape mismatches (mbs RAW)
        print("=== Check shape mismatches: mbs RAW export ===")
        cmd_check_mbs_raw = (
            f"python check_shape_mismatches.py "
            f"--path \"{RAW_DIR_MBS}\" "
            f"--img_folder \"imgs\" "
            f"--mask_folder \"masks\" "
            f"--mask_suffix \"_bolus\""
        )
        run_command(cmd_check_mbs_raw)
    else:
        print("Skipping Labelbox export for mbs (and RAW mismatch checks).")

    # 3) Prune the mbss_martin export
    print("=== 3. Prune the mbss_martin export ===")
    cmd_prune_mbss = (
        f"python preprocess_labelbox_export.py "
        f"--original_dataset_dir \"{RAW_DIR_MBSS}\" "
        f"--curation_csv \"{CURATION_CSV}\" "
        f"--output_dataset_dir \"{PRUNED_MBSS}\" "
        f"--verbose"
    )
    run_command(cmd_prune_mbss)

    # Check shape mismatches (mbss_martin pruned)
    print("=== Check shape mismatches: mbss_martin PRUNED export ===")
    cmd_check_mbss_pruned = (
        f"python check_shape_mismatches.py "
        f"--path \"{PRUNED_MBSS}\" "
        f"--img_folder \"imgs\" "
        f"--mask_folder \"masks\" "
        f"--mask_suffix \"_bolus\""
    )
    run_command(cmd_check_mbss_pruned)

    # 4) Prune the mbs export
    print("=== 4. Prune the mbs export ===")
    cmd_prune_mbs = (
        f"python preprocess_labelbox_export.py "
        f"--original_dataset_dir \"{RAW_DIR_MBS}\" "
        f"--curation_csv \"{CURATION_CSV}\" "
        f"--output_dataset_dir \"{PRUNED_MBS}\" "
        f"--verbose"
    )
    run_command(cmd_prune_mbs)

    # Check shape mismatches (mbs pruned)
    print("=== Check shape mismatches: mbs PRUNED export ===")
    cmd_check_mbs_pruned = (
        f"python check_shape_mismatches.py "
        f"--path \"{PRUNED_MBS}\" "
        f"--img_folder \"imgs\" "
        f"--mask_folder \"masks\" "
        f"--mask_suffix \"_bolus\""
    )
    run_command(cmd_check_mbs_pruned)

    # 5) Merge the pruned exports into a single dataset
    print("=== 5. Merge the pruned exports into a single dataset ===")
    cmd_merge = (
        f"python create_dataset_from_labelboxexports.py "
        f"--input_paths \"{PRUNED_MBSS}\" \"{PRUNED_MBS}\" "
        f"--output_path \"{FINAL_DATASET_PATH}\""
    )
    run_command(cmd_merge)

    # 6) Fix / Sanitize masks in the merged dataset
    print("=== 6. Fix / Sanitize masks in the merged dataset ===")
    sanitizer_cmd = (
        f"python mask_sanitizer.py "
        f"-p \"{PATH_PREFIX}/processed/{FINAL_DATASET_NAME}/masks\" "
        #f"-fix"
    )
    run_command(sanitizer_cmd)

    # 7) Resize frames and masks to 512x512 with padding
    print("=== 7. Resize frames and masks to 512x512 with padding ===")
    resize_cmd = (
        f"python resize_images.py "
        f"-p \"{PATH_PREFIX}/processed/{FINAL_DATASET_NAME}\" "
        f"--folders imgs masks "
        f"-size 512 "
        f"-m pad_resize "
        f"--in_place"
    )
    run_command(resize_cmd)

    # 8) Visualize a few sample frames + masks
    print("=== 8. Visualize a few sample frames + masks ===")
    visualize_cmd = (
        f"python visualize_dataset.py "
        f"-p \"{PATH_PREFIX}/processed/{FINAL_DATASET_NAME}\" "
        f"-n 20 "
        f"--mask_suffix _bolus"
    )
    run_command(visualize_cmd)

    # 9) Split into train/val/test sets
    print("=== 9. Split into train/val/test sets ===")
    split_output = f"{PATH_PREFIX}/processed/{FINAL_DATASET_NAME}_final"
    split_cmd = (
        f"python create_train_test_split.py "
        f"--input_dirs \"{FINAL_DATASET_PATH}\" "
        f"--output_dir \"{split_output}\""
    )
    run_command(split_cmd)

    print("=== Pipeline completed successfully! ===")

if __name__ == "__main__":
    main()
