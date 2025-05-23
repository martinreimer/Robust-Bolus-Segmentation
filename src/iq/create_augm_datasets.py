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
        if transform is not None:
            augmented = transform(image=image_input)["image"].squeeze()
        else:
            augmented = image
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

    ax1.plot(x, summary_df["psnr"], marker='o', label="PSNR (0-70)", color='tab:blue')
    ax1.plot(x, summary_df["niqe"], marker='^', label="NIQE (0-70)", color='tab:orange')
    ax2.plot(x, summary_df["ssim"], marker='s', label="SSIM (0-1)", color='tab:green')

    ax1.set_xlabel("Degradation Parameter")
    ax1.set_ylabel("PSNR / NIQE", color='black')
    ax1.set_ylim(0, 70)
    ax2.set_ylabel("SSIM", color='black')
    ax2.set_ylim(0, 1)

    #ax1.set_xticks(range(len(x)))
    # set x-ticks as numbers from 0-10
    x_values = np.arange(len(x))
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels(x_values, rotation=0)
    ax1.grid(True)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    #plt.title(f"Image Quality Assesment: {title_suffix}")
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
    '''
    MotionBlur
    | Parameter         | What it does                                                                |
    | ----------------- | --------------------------------------------------------------------------- |
    | `blur_limit`      | Kernel size (length of the motion line). Bigger = more motion.              |
    | `angle_range`     | Direction of motion in degrees. 0° = horizontal, 90° = vertical.            |
    | `direction_range` | Bias of the blur. `-1.0 = backward`, `0 = symmetric`, `1.0 = forward only`. |
    | `allow_shifted`   | Allows the kernel to be off-center (more realistic but less symmetric).     |

    '''

    import numpy as np
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform

    class PoissonNoise(ImageOnlyTransform):
        def __init__(self, scale: float = 1.0, p: float = 0.5):
            super().__init__(p)
            self.scale = scale  # Controls severity — higher = more noise

        def apply(self, img, **params):
            noisy = np.random.poisson(img.astype(np.float32) * self.scale) / self.scale
            return np.clip(noisy, 0, 255).astype(np.uint8)

        def get_transform_init_args_names(self):
            return ("scale",)

    class FixedContrast(ImageOnlyTransform):
        def __init__(self, alpha: float, p: float = 1.0):
            super().__init__(p)
            self.alpha = alpha  # Contrast factor (1 = original)

        def apply(self, img, **params):
            img = img.astype(np.float32)
            mean = np.mean(img)
            return np.clip((img - mean) * self.alpha + mean, 0, 255).astype(np.uint8)

        def get_transform_init_args_names(self):
            return ("alpha",)

    class FixedBrightness(ImageOnlyTransform):
        def __init__(self, alpha: float, p: float = 1.0):
            super().__init__(p)
            self.alpha = alpha  # Brightness multiplier (1.0 = no change)

        def apply(self, img, **params):
            img = img.astype(np.float32)
            return np.clip(img * self.alpha, 0, 255).astype(np.uint8)

        def get_transform_init_args_names(self):
            return ("alpha",)

    import cv2
    import numpy as np
    from albumentations.core.transforms_interface import ImageOnlyTransform

    class IsotropicDownsample(ImageOnlyTransform):
        def __init__(self, scale_factor: float, interpolation=cv2.INTER_LINEAR, p: float = 1.0):
            super().__init__(p)
            self.scale_factor = scale_factor
            self.interpolation = interpolation

        def apply(self, img, **params):
            h, w = img.shape[:2]
            new_w = max(1, int(w * self.scale_factor))
            new_h = max(1, int(h * self.scale_factor))
            downsampled = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
            upsampled = cv2.resize(downsampled, (w, h), interpolation=self.interpolation)
            return upsampled

        def get_transform_init_args_names(self):
            return ("scale_factor", "interpolation")

    degradation_configs = {
        "GaussianBlur": {
            "severities": [
                {"sigma_limit": (0.0, 0.0), "blur_limit": (1, 1), "p": 0},
                {"sigma_limit": (3.0, 3.0), "blur_limit": (7, 7), "p": 1},
                {"sigma_limit": (6.0, 6.0), "blur_limit": (13, 13), "p": 1},
                {"sigma_limit": (9.0, 9.0), "blur_limit": (19, 19), "p": 1},
                {"sigma_limit": (12.0, 12.0), "blur_limit": (25, 25), "p": 1},
                {"sigma_limit": (15.0, 15.0), "blur_limit": (31, 31), "p": 1},
                {"sigma_limit": (18.0, 18.0), "blur_limit": (37, 37), "p": 1},
                {"sigma_limit": (21.0, 21.0), "blur_limit": (43, 43), "p": 1},
                {"sigma_limit": (24.0, 24.0), "blur_limit": (49, 49), "p": 1},
                {"sigma_limit": (27.0, 27.0), "blur_limit": (55, 55), "p": 1},
                {"sigma_limit": (30.0, 30.0), "blur_limit": (61, 61), "p": 1},
            ],
            "transform_fn": lambda cfg: A.GaussianBlur(
                sigma_limit=cfg["sigma_limit"],
                blur_limit=cfg["blur_limit"],
                p=cfg["p"]
            )
        },

        "MotionBlur": {
            "severities": [
                {"blur_limit": (1, 1), "angle_range": (0, 0), "direction_range": (0.0, 0.0), "allow_shifted": False,
                 "p": 0},
                {"blur_limit": (13, 13), "angle_range": (0, 36), "direction_range": (-0.1, 0.1), "allow_shifted": False,
                 "p": 1.0},
                {"blur_limit": (25, 25), "angle_range": (0, 72), "direction_range": (-0.2, 0.2), "allow_shifted": False,
                 "p": 1.0},
                {"blur_limit": (37, 37), "angle_range": (0, 108), "direction_range": (-0.3, 0.3),
                 "allow_shifted": False, "p": 1.0},
                {"blur_limit": (49, 49), "angle_range": (0, 144), "direction_range": (-0.4, 0.4),
                 "allow_shifted": False, "p": 1.0},
                {"blur_limit": (61, 61), "angle_range": (0, 180), "direction_range": (-0.5, 0.5),
                 "allow_shifted": False, "p": 1.0},
                {"blur_limit": (73, 73), "angle_range": (0, 216), "direction_range": (-0.6, 0.6),
                 "allow_shifted": False, "p": 1.0},
                {"blur_limit": (85, 85), "angle_range": (0, 252), "direction_range": (-0.7, 0.7), "allow_shifted": True,
                 "p": 1.0},
                {"blur_limit": (97, 97), "angle_range": (0, 288), "direction_range": (-0.8, 0.8), "allow_shifted": True,
                 "p": 1.0},
                {"blur_limit": (109, 109), "angle_range": (0, 324), "direction_range": (-0.9, 0.9),
                 "allow_shifted": True, "p": 1.0},
                {"blur_limit": (121, 121), "angle_range": (0, 360), "direction_range": (-1.0, 1.0),
                 "allow_shifted": True, "p": 1.0},
            ],
            "transform_fn": lambda cfg: A.MotionBlur(
                blur_limit=cfg["blur_limit"],
                angle_range=cfg["angle_range"],
                direction_range=cfg["direction_range"],
                allow_shifted=cfg["allow_shifted"],
                p=cfg["p"]
            )
        },
        "IsotropicDownsampling": {
            "severities": [
                {"scale_factor": 0.95, "p": 0},
                {"scale_factor": 0.9, "p": 1.0},
                {"scale_factor": 0.8, "p": 1.0},
                {"scale_factor": 0.7, "p": 1.0},
                {"scale_factor": 0.6, "p": 1.0},
                {"scale_factor": 0.5, "p": 1.0},
                {"scale_factor": 0.4, "p": 1.0},
                {"scale_factor": 0.3, "p": 1.0},
                {"scale_factor": 0.2, "p": 1.0},
                {"scale_factor": 0.1, "p": 1.0},
                {"scale_factor": 0.0, "p": 1.0},
            ],
            "transform_fn": lambda cfg: IsotropicDownsample(
                scale_factor=cfg["scale_factor"],
                interpolation=cv2.INTER_LINEAR,
                p=cfg["p"]
            )
        },
        "GaussianNoise": {
            "severities": [
                {"var_limit": (0.0, 0.0), "p": 0},
                {"var_limit": (0.005, 0.005), "p": 1.0},
                {"var_limit": (0.01, 0.01), "p": 1.0},
                {"var_limit": (0.015, 0.015), "p": 1.0},
                {"var_limit": (0.02, 0.02), "p": 1.0},
                {"var_limit": (0.025, 0.025), "p": 1.0},
                {"var_limit": (0.03, 0.03), "p": 1.0},
                {"var_limit": (0.035, 0.035), "p": 1.0},
                {"var_limit": (0.04, 0.04), "p": 1.0},
                {"var_limit": (0.045, 0.045), "p": 1.0},
                {"var_limit": (0.05, 0.05), "p": 1.0},
            ],
            "transform_fn": lambda cfg: A.GaussNoise(
                cfg["var_limit"],
                p=cfg["p"]
            )
        },
        "PoissonNoise": {
            "severities": [
                {"scale": 128.0, "p": 0},  # Level 0: no noise
                {"scale": 128.0, "p": 1.0},  # Level 1
                {"scale": 64.0, "p": 1.0},  # Level 2
                {"scale": 32.0, "p": 1.0},  # Level 3
                {"scale": 16.0, "p": 1.0},  # Level 4
                {"scale": 8.0, "p": 1.0},  # Level 5
                {"scale": 4.0, "p": 1.0},  # Level 6
                {"scale": 2.0, "p": 1.0},  # Level 7
                {"scale": 1.0, "p": 1.0},  # Level 8
                {"scale": 0.5, "p": 1.0},  # Level 9
                {"scale": 0.25, "p": 1.0},  # Level 10: very noisy
            ],
            "transform_fn": lambda cfg: PoissonNoise(
                scale=cfg["scale"],
                p=cfg["p"]
            )
        },

        'Contrast': {
            "severities": [
                {"alpha": 2.0, "p": 1.0},
                {"alpha": 1.75, "p": 1.0},
                {"alpha": 1.5, "p": 1.0},
                {"alpha": 1.25, "p": 1.0},
                {"alpha": 1.0, "p": 1.0},
                {"alpha": 0.8, "p": 1.0},
                {"alpha": 0.6, "p": 1.0},
                {"alpha": 0.4, "p": 1.0},
                {"alpha": 0.25, "p": 1.0},
                {"alpha": 0.1, "p": 1.0},
            ],
            "transform_fn": lambda cfg: FixedContrast(
                alpha=cfg["alpha"],
                p=cfg["p"]
            )
        },
        'ContrastCompression': {
            "severities": [
                {"alpha": 0.95, "p": 0},
                {"alpha": 0.9, "p": 1.0},
                {"alpha": 0.8, "p": 1.0},
                {"alpha": 0.7, "p": 1.0},
                {"alpha": 0.6, "p": 1.0},
                {"alpha": 0.5, "p": 1.0},
                {"alpha": 0.4, "p": 1.0},
                {"alpha": 0.3, "p": 1.0},
                {"alpha": 0.2, "p": 1.0},
                {"alpha": 0.1, "p": 1.0},
                {"alpha": 0.0, "p": 1.0},
            ],
            "transform_fn": lambda cfg: FixedContrast(alpha=cfg["alpha"], p=cfg["p"])
        },
        'ContrastExpansion': {
            "severities": [
                {"alpha": 1, "p": 0},
                {"alpha": 2, "p": 1.0},
                {"alpha": 3, "p": 1.0},
                {"alpha": 4, "p": 1.0},
                {"alpha": 5, "p": 1.0},
                {"alpha": 6, "p": 1.0},
                {"alpha": 7, "p": 1.0},
                {"alpha": 8, "p": 1.0},
                {"alpha": 9, "p": 1.0},
                {"alpha": 10, "p": 1.0},
                {"alpha": 11, "p": 1.0},
            ],
            "transform_fn": lambda cfg: FixedContrast(alpha=cfg["alpha"], p=cfg["p"])
        },
        'BrightnessDimming': {
            "severities": [
                {"alpha": 1.0, "p": 1.0},  # Level 0: no dimming
                {"alpha": 0.9, "p": 1.0},  # Level 1: barely dim
                {"alpha": 0.8, "p": 1.0},
                {"alpha": 0.7, "p": 1.0},
                {"alpha": 0.6, "p": 1.0},
                {"alpha": 0.5, "p": 1.0},
                {"alpha": 0.4, "p": 1.0},
                {"alpha": 0.3, "p": 1.0},
                {"alpha": 0.2, "p": 1.0},
                {"alpha": 0.1, "p": 1.0},
                {"alpha": 0.0, "p": 1.0},  # Level 10: extremely dark
            ],
            "transform_fn": lambda cfg: FixedBrightness(alpha=cfg["alpha"], p=cfg["p"])
        },
        'BrightnessBoosting':{
            "severities": [
                {"alpha": 1.0, "p": 1.0},   # Level 0: no boost
                {"alpha": 2.5, "p": 1.0},
                {"alpha": 4, "p": 1.0},
                {"alpha": 5.5, "p": 1.0},
                {"alpha": 7, "p": 1.0},
                {"alpha": 8.5, "p": 1.0},
                {"alpha": 10, "p": 1.0},
                {"alpha": 11.5, "p": 1.0},
                {"alpha": 13.0, "p": 1.0},
                {"alpha": 14.5, "p": 1.0},
                {"alpha": 16, "p": 1.0},   # Level 10: extremely overexposed
            ],
            "transform_fn": lambda cfg: FixedBrightness(alpha=cfg["alpha"], p=cfg["p"])
        },
        "JPEGCompression": {
            "severities": [
                {"quality_range": (100, 100), "p": 0},
                {"quality_range": (90, 90), "p": 1.0},
                {"quality_range": (80, 80), "p": 1.0},
                {"quality_range": (70, 70), "p": 1.0},
                {"quality_range": (60, 60), "p": 1.0},
                {"quality_range": (50, 50), "p": 1.0},
                {"quality_range": (40, 40), "p": 1.0},
                {"quality_range": (30, 30), "p": 1.0},
                {"quality_range": (20, 20), "p": 1.0},
                {"quality_range": (10, 10), "p": 1.0},
                {"quality_range": (1, 1), "p": 1.0},
            ],
            "transform_fn": lambda cfg: A.ImageCompression(
                quality_range=cfg["quality_range"],
                p=cfg["p"]
            )
        },
        "ContrastCompression_BrightnessDimming": {
            "severities": [
                {"contrast": {"alpha": 0.95, "p": 1.0}, "brightness": {"alpha": 0.95, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.8, "p": 1.0}, "brightness": {"alpha": 0.8, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.6, "p": 1.0}, "brightness": {"alpha": 0.6, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.4, "p": 1.0}, "brightness": {"alpha": 0.4, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.2, "p": 1.0}, "brightness": {"alpha": 0.2, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.1, "p": 1.0}, "brightness": {"alpha": 0.1, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.08, "p": 1.0}, "brightness": {"alpha": 0.08, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.05, "p": 1.0}, "brightness": {"alpha": 0.05, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.03, "p": 1.0}, "brightness": {"alpha": 0.03, "p": 1.0}, "p": 1.0},
                {"contrast": {"alpha": 0.01, "p": 1.0}, "brightness": {"alpha": 0.01, "p": 1.0}, "p": 1.0},
            ],
            "transform_fn": lambda cfg: A.Compose([
                FixedContrast(alpha=cfg["contrast"]["alpha"], p=cfg["contrast"]["p"]),
                FixedBrightness(alpha=cfg["brightness"]["alpha"], p=cfg["brightness"]["p"]),
            ])
        },
        "MotionBlur_JPEG_ContrastNoise": {
            "severities": [
                {
                    "motion": {"blur_limit": (3, 5), "angle_range": (0, 10), "direction_range": (-0.1, 0.1), "p": 1.0},
                    "jpeg": {"quality_range": (90, 90), "p": 1.0},
                    "contrast": {"alpha": 0.8, "p": 1.0},
                    "noise": {"var_limit": (0.001, 0.002), "p": 1.0},
                    "p": 1,
                },
                {
                    "motion": {"blur_limit": (7, 9), "angle_range": (0, 45), "direction_range": (-0.3, 0.3), "p": 1.0},
                    "jpeg": {"quality_range": (70, 70), "p": 1.0},
                    "contrast": {"alpha": 0.4, "p": 1.0},
                    "noise": {"var_limit": (0.01, 0.02), "p": 1.0},
                    "p": 1,
                },
                {
                    "motion": {"blur_limit": (11, 15), "angle_range": (0, 90), "direction_range": (-0.5, 0.5), "p": 1.0},
                    "jpeg": {"quality_range": (50, 50), "p": 1.0},
                    "contrast": {"alpha": 0.2, "p": 1.0},
                    "noise": {"var_limit": (0.02, 0.5), "p": 1.0},
                    "p": 1,
                },
                {
                    "motion": {"blur_limit": (17, 21), "angle_range": (0, 180), "direction_range": (-0.8, 0.8), "p": 1.0},
                    "jpeg": {"quality_range": (40, 40), "p": 1.0},
                    "contrast": {"alpha": 0.1, "p": 1.0},
                    "noise": {"var_limit": (0.05, 0.1), "p": 1.0},
                    "p": 1,
                },
                {
                    "motion": {"blur_limit": (21, 25), "angle_range": (0, 360), "direction_range": (-1.0, 1.0), "p": 1.0},
                    "jpeg": {"quality_range": (30, 30), "p": 1.0},
                    "contrast": {"alpha": 0.05, "p": 1.0},
                    "noise": {"var_limit": (0.1, 0.15), "p": 1.0},
                    "p": 1,
                },
            ],
            "transform_fn": lambda cfg: A.Compose([
                degradation_configs["MotionBlur"]["transform_fn"](cfg["motion"]),
                degradation_configs["JPEGCompression"]["transform_fn"](cfg["jpeg"]),
                FixedContrast(alpha=cfg["contrast"]["alpha"], p=cfg["contrast"]["p"]),
                degradation_configs["GaussianNoise"]["transform_fn"](cfg["noise"]),
            ])
        }
    }

    # blur
    degradation_method = 'JPEGCompression'#"RandomMotionAffine"  # could be passed via CLI
    degrad_method_output_dir = base_output_dir / degradation_method
    quality_scores_summary, first_image_info = [], []

    config = degradation_configs[degradation_method]

    for i, severity_cfg in enumerate(config["severities"], start=0):
        transform_name = f"level_{i}"
        output_dir = degrad_method_output_dir / transform_name

        if severity_cfg["p"] == 0: # no augmentation
            transform = None
        else:
            transform = A.Compose([config["transform_fn"](severity_cfg)])
        apply_augmentation(args.input_dir, output_dir, transform, transform_name, image_paths, quality_scores_summary)

        first_img_path = output_dir / "imgs" / image_paths[0].name
        first_image_info.append((first_img_path, transform_name))

    quality_summary(quality_scores_summary, degrad_method_output_dir, title_suffix="degradation_method")
    #not working propertly: plot_first_images(first_image_info, degrad_method_output_dir / "first_frames_grid.png")

    # noise
    '''
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
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple augmented test sets for evaluation with quality metrics")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the base dataset with imgs/ and masks/")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save augmented test sets")
    parser.add_argument("--sample", type=int, default=None, help="Number of images to sample (default: all)")
    args = parser.parse_args()
    main(args)


r'''
python create_augm_datasets.py --input-dir "D:/Martin/thesis/data/processed/dataset_normal_0514_final_roi_crop/val" --output-dir "D:/Martin/thesis/data/iqa/augm/" --sample 30  
'''