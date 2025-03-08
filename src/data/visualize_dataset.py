"""
Script to overlay ground truth and prediction masks onto images for dataset sanity checks.
It randomly selects a given number of images, overlays the masks, and saves the results
in an 'overlays' folder inside the dataset directory.

python visualize_dataset.py -p D:\Martin\thesis\data\processed\dataset_0227_final\train -n 1500 --mask_suffix _bolus

python visualize_dataset.py -p D:\Martin\thesis\data\processed\dataset_train_val_test_split\train -n 20 --mask_suffix _bolus
"""


import argparse
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def overlay_mask_on_image(img_pil: Image.Image,
                          mask: np.ndarray,
                          mask_color=(255, 0, 255),  # Purple
                          alpha=0.2):
    """
    Overlay a ground truth mask on the original image.
    """
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    base_np = np.array(img_pil)

    mask_overlay = np.zeros_like(base_np)
    try:
        mask_overlay[mask != 0] = mask_color
    except:
        print(f"Error: Mask shape {mask.shape} does not match image shape {base_np.shape}")
        return None

    result = (alpha * mask_overlay + (1 - alpha) * base_np).astype(np.uint8)
    return Image.fromarray(result)


def process_images(dataset_path, img_folder='imgs', mask_folder='masks', num_samples=20, mask_suffix=''):
    dataset_path = Path(dataset_path)
    img_path = dataset_path / img_folder
    mask_path = dataset_path / mask_folder
    overlay_path = dataset_path / 'overlays'
    overlay_path.mkdir(parents=True, exist_ok=True)

    # Get image and mask filenames
    img_files = sorted(list(img_path.glob('*.png')) + list(img_path.glob('*.jpg')))
    mask_files = sorted(list(mask_path.glob(f'*{mask_suffix}.png')) + list(mask_path.glob(f'*{mask_suffix}.jpg')))

    if len(img_files) == 0 or len(mask_files) == 0:
        print("No images or masks found in the specified directories.")
        return

    # Randomly sample images if needed
    num_samples = min(num_samples, len(img_files))
    sampled_files = random.sample(img_files, num_samples)

    for img_file in sampled_files:
        print(f"Processing: {img_file.name}")
        mask_file = mask_path / (img_file.stem + mask_suffix + img_file.suffix)
        if not mask_file.exists():
            print(f"Warning: No mask found for {img_file.name}, skipping.")
            continue

        # Load image and mask
        img_pil = Image.open(img_file)
        if img_pil.mode != 'L':
            img_pil = img_pil.convert('L')
        gt_mask = np.array(Image.open(mask_file))

        overlayed_img = overlay_mask_on_image(img_pil.convert('RGB'), gt_mask)

        # Create side-by-side comparison plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].imshow(img_pil, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(overlayed_img)
        ax[1].set_title('Overlayed Image')
        ax[1].axis('off')

        plt.tight_layout()
        plt.savefig(overlay_path / f"overlay_{img_file.stem}.png")
        plt.close()
        print(f"Saved overlay: {overlay_path / f'overlay_{img_file.stem}.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay ground truth masks on images.')
    parser.add_argument('-p', '--path', required=True, help='Path to the dataset folder')
    parser.add_argument('--folders', nargs=2, default=['imgs', 'masks'], help='Subfolder names for images and masks')
    parser.add_argument('-n', '--num_samples', type=int, default=20, help='Number of images to process')
    parser.add_argument('--mask_suffix', type=str, default='', help='Optional suffix for mask files (e.g. "_bolus")')
    args = parser.parse_args()

    process_images(args.path, img_folder=args.folders[0], mask_folder=args.folders[1], num_samples=args.num_samples,
                   mask_suffix=args.mask_suffix)
