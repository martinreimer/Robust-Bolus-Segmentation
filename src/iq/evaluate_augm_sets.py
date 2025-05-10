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

        print(f"Processing: {img_path.name}")

        if not mask_path.exists():
            print(f"Missing GT mask for {image_id}, skipping.")
            continue

        img = Image.open(img_path).convert("L")
        gt = Image.open(mask_path).convert("L")
        gt_mask = (np.array(gt) > 128).astype(np.uint8)

        masks_dict = predict_img(model, img, device=device, thresholds=[threshold])
        pred_mask = masks_dict[threshold]

        print(f"Predicted mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")

        dice = dice_coefficient(pred_mask, gt_mask)
        print(f"Dice score: {dice:.4f}")

        dice_scores.append({"image_id": image_id, "dice_score": dice})

        # Save binary mask
        bin_mask = mask_to_image(pred_mask, mask_values)
        bin_mask.save(preds_dir / f"{image_id}_mask.png")

        # Save visualization
        overlay_gt = overlay_prediction_on_image(img, gt_mask, color=(0, 255, 0), alpha=0.3)
        overlay_pred = overlay_prediction_on_image(img, pred_mask, color=(255, 0, 255), alpha=0.3)
        fig = create_triple_plot(img, overlay_gt, overlay_pred)
        fig.save(viz_dir / f"{image_id}_viz.png")

    # Save per-frame dice scores as CSV
    df = pd.DataFrame(dice_scores)
    dice_path = testset_path / "dice_scores.csv"
    df.to_csv(dice_path, index=False)
    print(f"Saved per-frame dice scores to {dice_path}")

    return df["dice_score"].mean()

def plot_summary(summary_csv_path, output_path, title):
    try:
        df = pd.read_csv(summary_csv_path)
        df["Param"] = df["Param"].astype(float)
        df["MeanDice"] = df["MeanDice"].astype(float)
        df_sorted = df.sort_values("Param")
        plt.figure(figsize=(8, 5))
        plt.ylim(0, 1)
        plt.plot(df_sorted["Param"], df_sorted["MeanDice"], marker='o')
        plt.xlabel("Augmentation Parameter")
        plt.ylabel("Mean Dice Score")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Failed to generate plot: {e}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device, mask_values = load_model(Path(args.model_path))

    base_dir = Path(args.base_dir)
    all_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    results = []
    image_info_list = []
    for d in all_dirs:
        param_value = str(d.name)

        mean_dice = evaluate_single_set(d, model, device, mask_values, threshold=args.threshold)
        results.append({"Dataset": d.name, "Param": param_value, "MeanDice": mean_dice})
        print(f"{d.name}: Mean Dice = {mean_dice:.4f}")

    # Save summary as CSV only
    df_summary = pd.DataFrame(results)
    summary_path_csv = base_dir / "summary_overview.csv"
    df_summary.to_csv(summary_path_csv, index=False)
    print(f"Saved summary CSV to {summary_path_csv}")

    # Create and save plot
    plot_path = base_dir / "dice_vs_param_plot.png"
    plot_summary(summary_path_csv, plot_path, args.plot_title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate multiple test sets with a segmentation model")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory containing test sets")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions")
    parser.add_argument("--plot-title", type=str, default="Dice Score vs. Augmentation Strength", help="Title for the result plot")
    args = parser.parse_args()
    main(args)



r'''
python evaluate_augm_sets.py -- base-dir "D:/Martin/thesis/data/iqa/augm" --model-path "D:\Martin\thesis\training_runs\U-Net\runs\dainty-tree-322\checkpoints\checkpoint_epoch18.pth" --regex ".*_lr(0.0001|0.00001|0.000001).*"



python -m iq.evaluate_augm_sets --base-dir "D:/Martin/thesis/data/iqa/augm" --model-path "D:\Martin\thesis\training_runs\U-Net\runs\dainty-tree-322\checkpoints\checkpoint_epoch18.pth" --regex '^augm_brightness_brightness_limit_([0-9.]+)$'"


'''