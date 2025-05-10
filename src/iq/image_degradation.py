import os
import argparse
import random
from pathlib import Path
import shutil

import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt

# Updated DEGRADATIONS using ReplayCompose wrappers
DEGRADATIONS = {
    "noise": A.ReplayCompose([
        A.GaussNoise(std_range=(0.01, 0.02), mean_range=(0, 0), p=1.0)
    ]),
    "brightness": A.ReplayCompose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1.0)
    ]),
    "contrast": A.ReplayCompose([
        A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.6, p=1.0)
    ]),
    "blur": A.ReplayCompose([
        A.GaussianBlur(blur_limit=(3, 12), p=1.0)
    ]),
    "compression": A.ReplayCompose([
        A.ImageCompression(quality_range=(20, 50), p=1.0)
    ]),
    "all": A.ReplayCompose([
        A.GaussNoise(std_range=(0.01, 0.02), mean_range=(0, 0), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.6, p=1.0),
        A.GaussianBlur(blur_limit=(3, 12), p=1.0),
        #A.ImageCompression(quality_range=(20, 50), p=1.0)
    ])
}

def clear_directory(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))


def process_images(input_dir, output_dir, transform, transform_desc, sample_size=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    viz_dir = output_dir / "viz"

    # Clear output and viz directories if they exist
    clear_directory(output_dir)
    clear_directory(viz_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Save transform description to a text file once per degradation
    desc_path = output_dir / "transform_description.txt"
    with open(desc_path, "w") as f:
        f.write(transform_desc)

    image_paths = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if sample_size:
        image_paths = random.sample(image_paths, min(sample_size, len(image_paths)))

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read {path}")
            continue

        image_input = image[..., np.newaxis]  # Albumentations expects 3D (H, W, C)
        result = transform(image=image_input)
        degraded = result["image"].squeeze()

        # Save degraded image
        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), degraded)

        # Generate side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Original")
        axes[1].imshow(degraded, cmap='gray')
        axes[1].set_title("Degraded")
        for ax in axes:
            ax.axis('off')

        viz_path = viz_dir / (path.stem + "_viz.png")
        plt.tight_layout()
        plt.savefig(viz_path, dpi=100)
        plt.close()

def main(args):
    for name, transform in DEGRADATIONS.items():
        if args.degradation and name not in args.degradation:
            continue
        output_subdir = os.path.join(args.output_dir, name)
        transform_inner = transform.transforms[0]  # Get the inner transform from ReplayCompose
        transform_desc = f"{transform_inner.__class__.__name__}({transform_inner.__dict__})"
        print(f"Applying {name} to {args.sample or 'all'} images...")
        process_images(args.input_dir, output_subdir, transform, transform_desc, args.sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Degrade grayscale images using Albumentations")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the input image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--degradation", nargs="+", choices=list(DEGRADATIONS.keys()),
                        help="List of degradations to apply (default: all)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of images to randomly sample (default: all)")

    args = parser.parse_args()
    main(args)

r'''

data dir: D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504_test_final_roi_crop\test\imgs
output dir: D:\Martin\thesis\data\iqa\first_test\

python image_degradation.py --input-dir "D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop/test/imgs" --output-dir "D:/Martin/thesis/data/iqa/first_test/" --sample 10 --degradation noise brightness contrast blur compression

'''