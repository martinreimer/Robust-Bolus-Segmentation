"""
Script to identify and correct non-binary mask images.

This script scans a folder of grayscale mask images to check if they contain pixel values
outside the binary range (0, 255). If any values between 1 and 254 are found, the script
corrects the mask by setting all values to 0.

Usage:
    fixing: python .\mask_sanitizer.py -p ../../data/processed/leonard_dataset/masks -fix
    just info: python .\mask_sanitizer.py -p ../../data/processed/leonard_dataset/masks
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def collect_image_paths(folder_path, suffix=None):
    """
    Collects and returns sorted image paths from a given folder with an optional suffix filter.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a valid directory.")

    image_files = sorted(folder.glob("*.png"))
    if suffix:
        image_files = [f for f in image_files if f.name.endswith(suffix)]
    return image_files


def check_and_correct_masks(masks_folder, fix_masks):
    """
    Checks if mask images are binary and corrects them by setting all values < 254 to 0.
    """
    masks = collect_image_paths(masks_folder, suffix="_bolus.png")
    non_binary_masks = []

    for mask_path in masks:
        with Image.open(mask_path) as img:
            arr = np.array(img.convert('L'), dtype=np.uint8)

        unique_values = np.unique(arr)
        if any(0 < val < 254 for val in unique_values):
            non_binary_masks.append(str(mask_path))
            print(f"Non-binary mask found: {mask_path}")

            if fix_masks:
                # Correct the mask by thresholding values
                corrected_arr = np.where(arr < 254, 0, 255).astype(np.uint8)
                corrected_img = Image.fromarray(corrected_arr)
                corrected_img.save(mask_path)
                print(f"Corrected mask saved: {mask_path}")

    if non_binary_masks:
        print("Non-binary masks found:")
        for mask in non_binary_masks:
            print(mask)
    else:
        print("All masks are binary.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and correct binary mask images.")
    parser.add_argument("--masks_folder", "-p", required=True, help="Path to the masks folder")
    parser.add_argument("--fix_masks", "-fix", action='store_true', help="Whether to fix non-binary masks")
    args = parser.parse_args()

    check_and_correct_masks(args.masks_folder, args.fix_masks)
