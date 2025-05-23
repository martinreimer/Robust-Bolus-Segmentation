import os
import re
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from models.u_net.predict import load_model, predict_img, overlay_prediction_on_image, create_triple_plot, mask_to_image

def dice_coefficient(pred, gt):
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (pred.sum() + gt.sum() + 1e-6)

def evaluate_single_set(testset_path, model, device, mask_values, threshold=0.5):
    imgs_dir = Path(testset_path) / "imgs"
    masks_dir = Path(testset_path) / "masks"
    preds_dir = Path(testset_path) / "predictions"
    viz_dir = preds_dir / "viz"
    preds_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    dice_scores = []
    all_filenames = sorted(list(imgs_dir.glob("*.png")))

    for img_path in tqdm(all_filenames, desc=f"Evaluating {testset_path.name}"):
        image_id = img_path.stem
        mask_path = masks_dir / f"{image_id}_bolus.png"

        #print(f"Processing: {img_path.name}")

        if not mask_path.exists():
            print(f"Missing GT mask for {image_id}, skipping.")
            continue

        img = Image.open(img_path).convert("L")
        gt = Image.open(mask_path).convert("L")
        gt_mask = (np.array(gt) > 128).astype(np.uint8)

        masks_dict = predict_img(model, img, device=device, thresholds=[threshold])
        pred_mask = masks_dict[threshold]

        #print(f"Predicted mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")

        dice = dice_coefficient(pred_mask, gt_mask)
        #print(f"Dice score: {dice:.4f}")

        dice_scores.append({"image_id": image_id, "dice_score": dice})

        # Save binary mask
        bin_mask = mask_to_image(pred_mask, mask_values)
        bin_mask.save(preds_dir / f"{image_id}_mask.png")

        # Save visualization
        img_gray = img.copy()  # Preserve grayscale for left panel
        overlay_gt = overlay_prediction_on_image(img.copy(), gt_mask, color=(0, 255, 0), alpha=0.3)
        overlay_pred = overlay_prediction_on_image(img.copy(), pred_mask, color=(255, 0, 255), alpha=0.3)
        fig = create_triple_plot(img_gray, overlay_gt, overlay_pred)

        fig.save(viz_dir / f"{image_id}_viz.png")

    # Save per-frame dice scores as CSV
    df = pd.DataFrame(dice_scores)
    dice_path = testset_path / "dice_scores.csv"
    df.to_csv(dice_path, index=False)
    #print(f"Saved per-frame dice scores to {dice_path}")

    return df




def plot_summary(summary_csv_path, output_path, title):
    try:
        df = pd.read_csv(summary_csv_path)
        df_sorted = df.sort_values("Severity")

        x = df_sorted["Severity"]
        y = df_sorted["DSC_mean"]
        y_std = df_sorted["DSC_std"]

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(0, 1)

        # Plot shaded std area
        ax.fill_between(x, y - y_std, y + y_std, color='lightblue', alpha=0.5, label="±1 STD")

        # Plot mean DSC line and points
        ax.plot(x, y, color='blue', marker='o', markersize=3, linewidth=1, label="Mean DSC")
        ax.scatter(x, y, s=30, color='blue')

        # Axis labels and ticks
        ax.set_ylabel("DSC", fontsize=8)
        ax.set_xlabel("Severity", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(x.astype(str), fontsize=8)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis='y', labelsize=8)

        # Grid
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='both', alpha=0.4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Failed to generate plot: {e}")




def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device, mask_values = load_model(Path(args.model_path))

    base_dir = Path(args.base_dir)
    all_base_dirs = [base_dir / folder for folder in args.folders]

    for test_set_path in tqdm(all_base_dirs):
        all_dirs = [d for d in test_set_path.iterdir() if d.is_dir()]

        print(f"Method Folder: {test_set_path}")
        results = []
        all_scores = []
        for d in all_dirs:
            if not d.is_dir():
                print(f"Skipping {d}, not a directory.")
                continue

            df_scores = evaluate_single_set(d, model, device, mask_values, threshold=args.threshold)
            df_scores["Param"] = d.name  # Add folder label for grouping
            all_scores.append(df_scores)

        df_all = pd.concat(all_scores, ignore_index=True)
        df_all.to_csv(base_dir / "all_dice_scores.csv", index=False)

        df_all["Severity"] = df_all["Param"].str.extract(r"(\d+)").astype(int)
        df_summary = df_all.groupby("Severity")["dice_score"].agg(["mean", "std"]).reset_index()
        df_summary.rename(columns={"mean": "DSC_mean", "std": "DSC_std"}, inplace=True)
        summary_path_csv = base_dir / "summary_overview.csv"
        df_summary.to_csv(summary_path_csv, index=False)


        # Save plot
        plot_path = test_set_path / "dice_vs_param_plot.png"
        plot_summary(summary_path_csv, plot_path, args.plot_title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate multiple test sets with a segmentation model")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory containing test sets")
    parser.add_argument("--folders", nargs='+', required=True, help="List of folder names to evaluate")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions")
    parser.add_argument("--plot-title", type=str, default="Dice Score vs. Augmentation Strength", help="Title for the result plot")
    args = parser.parse_args()
    main(args)



r'''
python -m iq.evaluate_augm_sets --base-dir "D:/Martin/thesis/data/iqa/augm" --folders BrightnessBoosting BrightnessDimming ContrastCompression ContrastExpansion gaussianblur GaussianNoise IsotropicDownsampling JPEGCompression MotionBlur PoissonNoise RandomElasticMotion RandomMotionAffine --model-path "D:\Martin\thesis\training_runs\U-Net\runs\deft-morning-516\checkpoints\checkpoint_epoch6.pth"      

python -m iq.evaluate_augm_sets --base-dir "D:/Martin/thesis/data/iqa/augm" --folders gaussianblur --model-path "D:\Martin\thesis\training_runs\U-Net\runs\deft-morning-516\checkpoints\checkpoint_epoch6.pth"      

'''