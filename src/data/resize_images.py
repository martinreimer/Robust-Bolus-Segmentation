"""
This script resizes images and masks in a specified dataset directory.

It processes image and mask folders separately, applying different resizing methods.
The script collects statistics about the image resolutions before resizing and
resizes the images based on predefined target dimensions provided via arguments.

Usage:
    stats + fix resize: python resize_images.py -p ../../data/processed/leonard_dataset/ -width 512 -height 512 --folders imgs masks
    only stats: python resize_images.py -p ../../data/processed/leonard_dataset/ -width 512 -height 512 --folders imgs masks --only_stats

    #    python resize_images.py -p ../../data/foreback/processed/train --folders imgs masks --only_stats
Default processing folders are 'imgs' and 'masks'.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

def collect_image_paths(folder_path):
    """
    Collects image paths from the specified folder.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a valid directory.")
    return sorted(folder.glob("*.png"))

def get_image_resolution(image_path):
    """
    Returns the resolution of the given image.
    """
    with Image.open(image_path) as img:
        return img.size

def resize_image(image_path, target_size, is_mask):
    """
    Resizes an image to the given target size.
    """
    img = Image.open(image_path)
    if is_mask:
        img = img.resize(target_size, Image.NEAREST)
    else:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    img.save(image_path)

def process_folder(folder_path, target_size, is_mask, only_stats):
    """
    Processes all images in the specified folder, optionally resizing them.
    """
    image_paths = collect_image_paths(folder_path)
    resolutions = [get_image_resolution(p) for p in image_paths]
    resolution_counts = Counter(resolutions)

    print(f"Before processing stats for {folder_path}:")
    for res, count in resolution_counts.items():
        print(f"Resolution {res}: {count} files")

    if not only_stats:
        for image_path in image_paths:
            resize_image(image_path, target_size, is_mask)
        print(f"Resizing complete for {folder_path}")

def main():
    parser = argparse.ArgumentParser(description="Resize images and masks in dataset folders.")
    parser.add_argument('--dataset_path', '-p', required=True, help="Path to the dataset directory")
    parser.add_argument('--target_width', '-width', type=int, required=False, default=512, help="Target width for resizing")
    parser.add_argument('--target_height', '-height', type=int, required=False, default=512, help="Target height for resizing")
    parser.add_argument('--folders', '-f', nargs='+', default=['imgs', 'masks'],
                        help="Folders to process (default: imgs masks)")
    parser.add_argument('--only_stats', action='store_true', help="Only collect resolution statistics without resizing")
    args = parser.parse_args()

    target_size = (args.target_width, args.target_height)

    for folder in args.folders:
        folder_path = Path(args.dataset_path) / folder
        if folder_path.exists():
            is_mask = 'masks' in folder.lower()
            print(f"Processing folder {folder_path} which is a mask folder: {is_mask}")
            process_folder(folder_path, target_size, is_mask, args.only_stats)
        else:
            print(f"Warning: Folder {folder_path} does not exist.")
    print("Processing complete.")

if __name__ == "__main__":
    main()
