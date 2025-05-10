import os
import argparse
import random
from pathlib import Path
import shutil
import numpy as np
import cv2
import albumentations as A
import pandas as pd
from sewar.full_ref import psnr, ssim
from PIL import Image
from niqe_score import niqe
import matplotlib.pyplot as plt



def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def apply_augmentation(input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary):
    imgs_dir = Path(input_dir) / "imgs"
    masks_dir = Path(input_dir) / "masks"
    out_imgs_dir = Path(output_dir) / "imgs"
    out_masks_dir = Path(output_dir) / "masks"

    clear_directory(out_imgs_dir)
    clear_directory(out_masks_dir)

    records = []

    for img_path in image_paths:

        image_id = img_path.name
        mask_name = f"{img_path.stem}_bolus.png"
        mask_path = masks_dir / mask_name

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue

        image_input = image[..., np.newaxis]
        augmented = transform(image=image_input)["image"].squeeze()

        # Save degraded image
        cv2.imwrite(str(out_imgs_dir / img_path.name), augmented)

        # Copy original mask
        if mask_path.exists():
            shutil.copy(mask_path, out_masks_dir / mask_path.name)
        else:
            print(f"Warning: Mask not found for {img_path.name}")

        # Clean images for NIQE
        #image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        #augmented = np.nan_to_num(augmented, nan=0.0, posinf=255.0, neginf=0.0)

        def clean_for_niqe(img):
            # Ensure float32, clean NaNs/infs, clip to valid range, and check shape
            img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
            img = np.clip(img, 0, 255).astype(np.float32)
            if img.shape[0] < 192 or img.shape[1] < 192:
                raise ValueError("Image must be at least 192x192 for NIQE.")
            return img

        # Compute quality metrics
        try:
            niqe_orig = niqe(clean_for_niqe(image))
            #niqe_orig = niqe(image)
        except Exception as e:
            niqe_orig = np.nan
            print(f"NIQE error (original) for {img_path.name}: {e}")
        try:
            niqe_degraded = niqe(clean_for_niqe(augmented))
        except Exception as e:
            niqe_degraded = np.nan
            print(f"NIQE error (degraded) for {img_path.name}: {e}")

        img_ref = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_deg = cv2.cvtColor(augmented, cv2.COLOR_GRAY2BGR)
        # check if both are different
        if np.array_equal(img_ref, img_deg):
            psnr_score = 100.0#np.inf
            ssim_score = 1.0
        else:
            try:
                psnr_score = psnr(img_ref, img_deg)
            except Exception as e:
                psnr_score = np.nan
                print(f"PSNR error for {img_path.name}: {e}")
            try:
                ssim_score = ssim(image, augmented)[0]
            except Exception as e:
                ssim_score = np.nan
                print(f"SSIM error for {img_path.name}: {e}")

        records.append({
            "image": image_id,
            "niqe_original": niqe_orig,
            "niqe_degraded": niqe_degraded,
            "psnr": psnr_score,
            "ssim": ssim_score
        })

    df = pd.DataFrame(records)
    # Clean infs before averaging
    df_cleaned = df.copy()
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert any tuples (e.g., from SSIM) to scalar if needed
    for col in df_cleaned.columns[1:]:
        df_cleaned[col] = df_cleaned[col].apply(lambda x: x[0] if isinstance(x, (tuple, list)) else x)

    # Compute mean safely
    avg_values = df_cleaned.iloc[:, 1:].mean(numeric_only=True)
    expected_cols = df_cleaned.shape[1] - 1
    avg_values = list(avg_values.values)
    avg_values += [np.nan] * (expected_cols - len(avg_values))

    # Append AVG row
    df.loc[len(df.index)] = ["AVG"] + avg_values

    df.to_csv(Path(output_dir) / "image_quality_scores.csv", index=False)

    avg_row = df[df["image"] == "AVG"].iloc[0]
    quality_scores_summary.append({
        "dataset": transform_name,
        "niqe": avg_row["niqe_degraded"],
        "psnr": avg_row["psnr"],
        "ssim": avg_row["ssim"],
    })

    with open(Path(output_dir) / "transform_description.txt", "w") as f:
        f.write(transform_name)

def plot_quality_summary(summary_df, output_path, title_suffix=""):
    #summary_df = summary_df.sort_index()
    x = summary_df.index.tolist()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(x, summary_df["psnr"], marker='o', label="PSNR (0-100)", color='tab:blue')
    ax1.plot(x, summary_df["niqe"], marker='^', label="NIQE (0-100)", color='tab:orange')
    ax2.plot(x, summary_df["ssim"], marker='s', label="SSIM (0-1)", color='tab:green')

    ax1.set_xlabel("Degradation Parameter")
    ax1.set_ylabel("PSNR / NIQE", color='black')
    ax1.set_ylim(0, 100)
    ax2.set_ylabel("SSIM", color='black')
    ax2.set_ylim(0, 1)

    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax1.grid(True)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f"Image Quality Assesment: {title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_first_images(image_info_list, output_path):
    try:
        num_images = len(image_info_list)
        fig, axs = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
        if num_images == 1:
            axs = [axs]
        for ax, (img_path, title) in zip(axs, image_info_list):
            img = Image.open(img_path)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{title}", fontsize=10)
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.1, 1, 0.9])
        plt.savefig(output_path)
        print(f"Saved first-frame grid plot to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Failed to generate first-frame plot: {e}")

def quality_summary(summary_list, output_path, title_suffix=""):
    # Save summary across test sets as CSV
    df_summary = pd.DataFrame(summary_list)
    df_summary.set_index("dataset", inplace=True)
    df_summary.to_csv(output_path / f"iq_summary_across_testsets_{title_suffix}.csv")
    # Plot summary
    plot_path = output_path / f"iq_summary_plot{title_suffix}.png"
    plot_quality_summary(df_summary, output_path, title_suffix=title_suffix)

def main(args):
    base_output_dir = Path(args.output_dir)
    imgs_dir = Path(args.input_dir) / "imgs"
    image_paths = list(imgs_dir.glob("*.png"))
    if args.sample:
        random.seed(42)
        image_paths = random.sample(image_paths, min(args.sample, len(image_paths)))

    '''
    # less brightness
    degrad_method_output_dir = base_output_dir / "less_brightness"
    quality_scores_summary, first_image_info = [], []
    for param in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-param, -param), contrast_limit=0.0, p=1.0)])
        transform_name = f"{param}"
        output_dir = degrad_method_output_dir / transform_name
        apply_augmentation(args.input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary)
        first_img_path = output_dir / "imgs" / image_paths[0].name
        first_image_info.append((first_img_path, f"{param}"))
    quality_summary(quality_scores_summary, degrad_method_output_dir, title_suffix="Less Brightness")
    plot_first_images(first_image_info, degrad_method_output_dir / "first_frames_grid.png")

    # more brightness
    degrad_method_output_dir = base_output_dir / "more_brightness"
    quality_scores_summary, first_image_info = [], []
    for param in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        transform = A.Compose(
            [A.RandomBrightnessContrast(brightness_limit=(param, param), contrast_limit=0.0, p=1.0)])
        transform_name = f"{param}"
        output_dir = degrad_method_output_dir / transform_name
        apply_augmentation(args.input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary)
        first_img_path = output_dir / "imgs" / image_paths[0].name
        first_image_info.append((first_img_path, f"{param}"))
    quality_summary(quality_scores_summary, degrad_method_output_dir, title_suffix="More Brightness")
    plot_first_images(first_image_info, degrad_method_output_dir / "first_frames_grid.png")

    # less contrast
    degrad_method_output_dir = base_output_dir / "less_contrast"
    quality_scores_summary, first_image_info = [], []
    for param in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=(-param, -param), p=1.0)])
        transform_name = f"{param}"
        output_dir = degrad_method_output_dir / transform_name
        apply_augmentation(args.input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary)
        first_img_path = output_dir / "imgs" / image_paths[0].name
        first_image_info.append((first_img_path, f"{param}"))
    quality_summary(quality_scores_summary, degrad_method_output_dir, title_suffix="Less Contrast")
    plot_first_images(first_image_info, degrad_method_output_dir / "first_frames_grid.png")

    # more contrast
    degrad_method_output_dir = base_output_dir / "more_contrast"
    quality_scores_summary, first_image_info = [], []
    for param in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        transform = A.Compose(
            [A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=(param, param), p=1.0)])
        transform_name = f"{param}"
        output_dir = degrad_method_output_dir / transform_name
        apply_augmentation(args.input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary)
        first_img_path = output_dir / "imgs" / image_paths[0].name
        first_image_info.append((first_img_path, f"{param}"))
    quality_summary(quality_scores_summary, degrad_method_output_dir, title_suffix="More Contrast")
    plot_first_images(first_image_info, degrad_method_output_dir / "first_frames_grid.png")
    '''
    # blur
    degrad_method_output_dir = base_output_dir / "blur"
    quality_scores_summary, first_image_info = [], []
    for param in [0, 2, 5, 10, 15, 30, 60, 90, 120, 150, 180, 210, 240, 270, 400, 600, 1000]:
        transform = A.Compose([A.GaussianBlur(blur_limit=(param, param), p=1.0)])
        transform_name = f"{param}"
        output_dir = degrad_method_output_dir / transform_name
        apply_augmentation(args.input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary)
        first_img_path = output_dir / "imgs" / image_paths[0].name
        first_image_info.append((first_img_path, f"{param}"))
    quality_summary(quality_scores_summary, degrad_method_output_dir, title_suffix="Blur")
    plot_first_images(first_image_info, degrad_method_output_dir / "first_frames_grid.png")

    # noise
    degrad_method_output_dir = base_output_dir / "noise"
    quality_scores_summary, first_image_info = [], []
    for param in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 10, 50, 100, 200]:
        transform = A.Compose([A.GaussNoise(var_limit=(param, param), mean=0, always_apply=True, p=1)])
        transform_name = f"{param}"
        output_dir = degrad_method_output_dir / transform_name
        apply_augmentation(args.input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary)
        first_img_path = output_dir / "imgs" / image_paths[0].name
        first_image_info.append((first_img_path, f"{param}"))
    quality_summary(quality_scores_summary, degrad_method_output_dir, title_suffix="Noise")
    plot_first_images(first_image_info, degrad_method_output_dir / "first_frames_grid.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple augmented test sets for evaluation with quality metrics")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the base dataset with imgs/ and masks/")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save augmented test sets")
    parser.add_argument("--sample", type=int, default=None, help="Number of images to sample (default: all)")
    args = parser.parse_args()
    main(args)


r'''
python create_augm_datasets.py --input-dir "D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop/test" --output-dir "D:/Martin/thesis/data/iqa/augm/" --sample 50  
'''