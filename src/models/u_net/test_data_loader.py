"""
Method for testing data loading in general and augmentations in particular.

Augmentation test: python .\test_data_loader.py --test-augmentations --num-samples 30
Simulate training epoch: python .\test_data_loader.py --simulate-training-epoch
"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import albumentations as A
import os
import warnings
import cv2
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Test DataLoader by plotting images and masks.")
    parser.add_argument('--dataset-path', '-d', type=str, default='D:/Martin/thesis/data/processed/dataset_0228_final',
                        help='Path to the dataset.')
    parser.add_argument('--subset', type=str, default='train', help='Subset to use: train, val, or test')
    parser.add_argument('--mask-suffix', type=str, default='_bolus', help='Suffix for mask files.')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to plot.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for DataLoader.')
    parser.add_argument('--test-augmentations', action='store_true', help='Test augmentations instead of raw images.')
    parser.add_argument('--simulate-training-epoch', action='store_true',
                        help='Simulate one training epoch by applying augmentations on all images and saving triplet plots.')
    return parser.parse_args()


def extract_transform_names(replay):
    """
    Extracts the names of applied transforms from the Albumentations replay dictionary.
    This function works for spatial transforms.

    Returns a string of the format:
      "Transform1, Transform2, ..."
    """
    if replay is None:
        return "None"
    transforms = replay.get("transforms", [])
    names = []
    for t in transforms:
        if t.get("applied", False):
            class_fullname = t.get("__class_fullname__", "Unknown")
            transform_name = class_fullname.split('.')[-1]
            names.append(transform_name)
    return ", ".join(names)


def extract_intensity_transform_names(replay):
    """
    Extracts the name of the applied intensity transform.
    If a OneOf is used, it digs into its transforms to get the actual transform applied.

    Returns a string, e.g. "RandomBrightnessContrast" or "GaussianBlur".
    """
    if replay is None:
        return "None"
    names = []
    for t in replay.get("transforms", []):
        if t.get("applied", False):
            class_fullname = t.get("__class_fullname__", "Unknown")
            if "OneOf" in class_fullname:
                # Look inside the OneOf container for the transform that was applied.
                inner_transforms = t.get("transforms", [])
                for inner in inner_transforms:
                    if inner.get("applied", False):
                        inner_name = inner.get("__class_fullname__", "Unknown").split('.')[-1]
                        names.append(inner_name)
            else:
                names.append(class_fullname.split('.')[-1])
    return ", ".join(names)


def get_augmentations():
    """
    Defines a two-stage augmentation pipeline using ReplayCompose.
    Spatial transforms are applied to both image and mask.
    Intensity transforms are applied only to the image.

    Returns:
        A function that accepts image and mask and returns:
        (augmented_image, augmented_mask, replay_spatial, replay_intensity)
    """
    # Spatial transforms (applied to both image and mask)
    spatial_transform = A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), shear=(-7, 7), p=0.7),
    ], additional_targets={'mask': 'mask'})

    # Intensity transforms (applied only to the image)
    intensity_transform = A.ReplayCompose([
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.Lambda(name="briiightnesse", image=lambda x, **kwargs: (x * 1.5).clip(0, 1).astype(x.dtype), p=1.0),
    ])

    def augment(image, mask):
        # Albumentations expects images with shape (H, W, C)
        spatial_out = spatial_transform(image=image[..., None], mask=mask[..., None])
        image_spatial = spatial_out['image'].squeeze(-1)
        mask_spatial = spatial_out['mask'].squeeze(-1)
        replay_spatial = spatial_out.get('replay', None)

        # Apply intensity transforms only to the image
        intensity_out = intensity_transform(image=image_spatial[..., None])
        image_final = intensity_out['image'].squeeze(-1)
        replay_intensity = intensity_out.get('replay', None)

        return image_final, mask_spatial, replay_spatial, replay_intensity

    return augment


def overlay_mask_on_image(image, mask, overlay_color=(0.5, 0, 0.5), alpha=0.4):
    """
    Overlays the binary mask on the image using a given color and alpha for blending.

    Args:
        image (np.ndarray): Grayscale image with shape (H, W) in range [0,1].
        mask (np.ndarray): Binary mask with shape (H, W). Nonzero values indicate the mask.
        overlay_color (tuple): RGB color for the overlay in normalized [0,1] values.
        alpha (float): Blending factor.

    Returns:
        np.ndarray: Image with mask overlay as RGB with shape (H, W, 3).
    """
    rgb_image = np.dstack([image, image, image])
    mask_bool = mask > 0
    overlay = rgb_image.copy()
    overlay[mask_bool] = (1 - alpha) * rgb_image[mask_bool] + alpha * np.array(overlay_color)
    return overlay


def test_augmentations(args):
    """
    Loads a few samples from the dataset, applies augmentations, and plots:
      - Original image
      - Augmented image (image only)
      - Augmented image with mask overlay in purple
    At the bottom of the triple plot, a one-liner displays the applied spatial and intensity transforms.
    """
    dataset = BasicDataset(base_dir=args.dataset_path, subset=args.subset,
                           mask_suffix=args.mask_suffix, transform=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    augment = get_augmentations()

    logging.info(f"Dataset size: {len(dataset)}")

    for idx, sample in enumerate(dataloader):
        original_image = sample['image'][0].squeeze(0).numpy()
        original_mask = sample['mask'][0].squeeze(0).numpy()

        aug_image, aug_mask, replay_spatial, replay_intensity = augment(original_image, original_mask)
        spatial_names = extract_transform_names(replay_spatial)
        intensity_names = extract_intensity_transform_names(replay_intensity)

        overlay_image = overlay_mask_on_image(aug_image, aug_mask, overlay_color=(0.5, 0, 0.5), alpha=0.4)

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        mng = plt.get_current_fig_manager()
        try:
            mng.window.setGeometry(0, 50, 1920, 1080)
        except AttributeError:
            try:
                mng.window.wm_geometry("+0+100")
            except Exception as e:
                print("Could not set window geometry:", e)

        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(aug_image, cmap='gray')
        axes[1].set_title('Augmented Image')
        axes[1].axis('off')

        axes[2].imshow(overlay_image)
        axes[2].set_title('Augmented Image with Mask Overlay')
        axes[2].axis('off')

        fig.text(0.5, 0.02, f"Spatial: {spatial_names} | Intensity: {intensity_names}", ha='center', fontsize=14)

        plt.suptitle(f"Sample {idx}", fontsize=16)
        plt.show()

        if idx + 1 >= args.num_samples:
            break


def test_raw_data(args):
    """
    Plots raw images and masks from the dataset.
    """
    dataset = BasicDataset(base_dir=args.dataset_path, subset=args.subset,
                           mask_suffix=args.mask_suffix, transform=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logging.info(f"Dataset size: {len(dataset)}")

    for idx, sample in enumerate(dataloader):
        image = sample['image'][0].squeeze(0).numpy()
        mask = sample['mask'][0].squeeze(0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.suptitle(f"Sample {idx}")
        plt.show()

        if idx + 1 >= args.num_samples:
            break


def simulate_training_epoch(args):
    """
    Simulates one training epoch by applying augmentations on all images in the dataset.
    For each sample, creates a triplet plot:
      - Original image
      - Augmented image (image only)
      - Augmented image with mask overlay in purple
    Saves each plot as a PNG file in a folder called 'test_augmentations' located in the script directory.
    """
    # Determine output directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "test_augmentations")
    os.makedirs(output_dir, exist_ok=True)

    dataset = BasicDataset(base_dir=args.dataset_path, subset=args.subset,
                           mask_suffix=args.mask_suffix, transform=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    augment = get_augmentations()

    logging.info(f"Simulating training epoch on dataset of size: {len(dataset)}")
    for idx, sample in enumerate(dataloader):
        original_image = sample['image'][0].squeeze(0).numpy()
        original_mask = sample['mask'][0].squeeze(0).numpy()

        aug_image, aug_mask, replay_spatial, replay_intensity = augment(original_image, original_mask)
        spatial_names = extract_transform_names(replay_spatial)
        intensity_names = extract_intensity_transform_names(replay_intensity)
        overlay_image = overlay_mask_on_image(aug_image, aug_mask, overlay_color=(0.5, 0, 0.5), alpha=0.4)

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        # Plot original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot augmented image
        axes[1].imshow(aug_image, cmap='gray')
        axes[1].set_title('Augmented Image')
        axes[1].axis('off')

        # Plot augmented image with mask overlay
        axes[2].imshow(overlay_image)
        axes[2].set_title('Augmented Image with Mask Overlay')
        axes[2].axis('off')

        # Add one-liner text at the bottom with augmentation details
        fig.text(0.5, 0.02, f"Spatial: {spatial_names} | Intensity: {intensity_names}", ha='center', fontsize=14)
        plt.suptitle(f"Sample {idx}", fontsize=16)

        # Save the figure to the output folder
        output_file = os.path.join(output_dir, f"sample_{idx:04d}.png")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved augmented sample {idx} to {output_file}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.simulate_training_epoch:
        simulate_training_epoch(args)
    elif args.test_augmentations:
        test_augmentations(args)
    else:
        test_raw_data(args)


if __name__ == '__main__':
    main()
