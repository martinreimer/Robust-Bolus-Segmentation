#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory.")
        return

    # We'll collect histogram counts for intensities 0..255
    combined_hist = np.zeros(256, dtype=np.int64)

    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ])

    if not image_files:
        print(f"No valid image files found in {folder}")
        return

    print(f"Found {len(image_files)} images in {folder}, building histogram...")

    first_img_100_200 = None  # Will store the first path that has [100..200]

    for img_path in image_files:
        # Load image in grayscale
        with Image.open(img_path) as img:
            gray_img = img.convert('L')
            arr = np.array(gray_img, dtype=np.uint8)

        # Update combined histogram for this image
        img_hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        combined_hist += img_hist

        # If we haven't picked a "first image with [100..200]" yet, check this one
        if first_img_100_200 is None:
            # If there's any pixel in [100..200]
            if ((arr >= 100) & (arr <= 200)).any():
                first_img_100_200 = img_path

    # --- 1) Show the histogram of intensities across all images ---
    plt.figure(figsize=(8, 4))
    plt.bar(range(256), combined_hist, width=1.0, color='blue')
    plt.title(f'Combined Grayscale Histogram (0..255)\n({len(image_files)} images in "{folder}")')
    plt.xlabel('Pixel intensity (0..255)')
    plt.ylabel('Count')
    plt.tight_layout()

    # --- 2) If we found at least one image with [100..200], color-code the first one ---
    if first_img_100_200 is not None:
        with Image.open(first_img_100_200) as img:
            gray_img = img.convert('L')
            arr = np.array(gray_img, dtype=np.uint8)

        # Create an RGB image from the grayscale, following the rules:
        #   - 0 => black = (0,0,0)
        #   - 1 => white = (255,255,255)
        #   - else => red with intensity => (val, 0, 0)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Vectorized approach:
        # mask for val=0 => black
        zero_mask = (arr == 0)
        # mask for val=1 => white
        one_mask = (arr == 1)
        # everything else => set in the red channel
        else_mask = (~zero_mask & ~one_mask)

        # Fill black
        rgb[zero_mask] = (0, 0, 0)
        # Fill white
        rgb[one_mask] = (255, 255, 255)

        # Fill red
        # The red channel is the original grayscale intensity
        # e.g. arr=128 => (128,0,0)
        red_vals = arr[else_mask]
        rgb[else_mask, 0] = red_vals  # put grayscale intensity in the red channel

        # Show color-coded image in a new figure
        plt.figure()
        plt.imshow(rgb)
        plt.title(f'First image with [100..200]: {first_img_100_200.name}')
        plt.axis('off')

    else:
        print("\nNo image has intensities in [100..200].")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build a combined grayscale histogram for all images in a folder. "
                    "Also pick first image with intensities in [100..200], color-code it, and show."
    )
    parser.add_argument('folder', type=str,
                        help='Path to the folder containing grayscale mask images (e.g. ./data/masks).')
    args = parser.parse_args()

    main(args.folder)
