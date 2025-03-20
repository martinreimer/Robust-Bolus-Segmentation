"""
Example:
Prediction w/ groud truth + save as mp4s + plot metrics also + save metrics as csv -> triple plot
python predict.py --test-name breezy-river-141 --model-path D:/Martin/thesis/training_runs/U-Net/runs/breezy-river-141/checkpoints/checkpoint_epoch25.pth --output-dir D:\Martin\thesis\test_runs --data-dir D:/Martin/thesis/data/processed/dataset_0228_final/test/ --csv-path D:/Martin/thesis/data/processed/dataset_0228_final/data_overview.csv -v -t 0.8 --save-metrics-csv --save-video-mp4s --plot-metrics --fps 10

Prediction w/ groud truth + save as mp4s + save metrics as csv -> double plot
python predict.py --test-name breezy-river-141 --model-path D:/Martin/thesis/training_runs/U-Net/runs/breezy-river-141/checkpoints/checkpoint_epoch25.pth --output-dir D:\Martin\thesis\test_runs --data-dir D:/Martin/thesis/data/processed/dataset_0228_final/test/ --csv-path D:/Martin/thesis/data/processed/dataset_0228_final/data_overview.csv -v -t 0.8 --save-metrics-csv --save-video-mp4s --fps 10


This script performs segmentation predictions using a pre-trained UNet model.
It supports different input modes:
  - Prediction only (without ground truth): use --no-gt.
  - Prediction only with frame-to-video mapping (via CSV): use --no-gt and --csv-path.
  - Prediction with ground truth: provide ground truth images (do not set --no-gt).

Output options include:
  - Saving individual frame plots (--save-frame-plots)
  - Creating a GIF per video (--save-video-gifs)
  - Creating an MP4 per video (--save-video-mp4s)
  - Saving metrics (BCE and Dice scores) as CSV (--save-metrics-csv) and as a TXT file.

Required arguments: model path, data directory, output directory, test name.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import pandas as pd
import imageio
from tqdm import tqdm

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask  # if needed

from moviepy.editor import ImageSequenceClip
# --------------------------
# Helper Functions
# --------------------------
def load_model(model_path, channels, classes, bilinear):
    logging.info("Building UNet...")
    net = UNet(n_channels=channels, n_classes=classes, bilinear=bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info("Model loaded successfully!")
    return net, device, mask_values

def predict_img(net, full_img: Image.Image, device: torch.device, scale_factor=1.0, thresholds=[0.5]):
    net.eval()
    full_img = full_img.convert('L') if net.n_channels == 1 else full_img.convert('RGB')
    img_np = np.array(full_img, dtype=np.float32)
    img_np = BasicDataset.preprocess(img_np, is_mask=False)
    if net.n_channels == 1 and img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=0)
    elif net.n_channels == 3 and img_np.ndim == 3:
        img_np = np.transpose(img_np, (2, 0, 1))
    img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img_torch).cpu()
        orig_size = (full_img.size[1], full_img.size[0])
        output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=False)
        masks = {}
        for threshold in thresholds:
            masks[threshold] = (torch.sigmoid(output) > threshold).long().squeeze().numpy()
    return masks

def mask_to_image(mask: np.ndarray, mask_values):
    if len(mask_values) == 2 and set(mask_values) == {0, 1}:
        out = np.zeros(mask.shape, dtype=np.uint8)
        out[mask == 1] = 255
        return Image.fromarray(out)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for i, val in enumerate(mask_values):
        out[mask == i] = val
    return Image.fromarray(out)

def overlay_prediction_on_image(img_pil: Image.Image, mask: np.ndarray, color=(255, 0, 255), alpha=0.3):
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_np = np.array(img_pil)
    overlay = np.zeros_like(img_np)
    overlay[mask != 0] = color
    result = (alpha * overlay + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(result)

def create_double_plot(original_img, overlay_pred_img, title_text=None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    axs[0].imshow(original_img)
    axs[0].set_title("Original", pad=5)
    axs[0].axis('off')
    axs[1].imshow(overlay_pred_img)
    axs[1].set_title("Prediction Overlay", pad=5)
    axs[1].axis('off')
    if title_text:
        fig.suptitle(title_text, y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)

def create_triple_plot(original_img, overlay_gt_img, overlay_pred_img, title_text=None):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), dpi=100)
    axs[0].imshow(original_img)
    axs[0].set_title("Original", pad=5)
    axs[0].axis('off')
    axs[1].imshow(overlay_gt_img)
    axs[1].set_title("Ground Truth Overlay", pad=5)
    axs[1].axis('off')
    axs[2].imshow(overlay_pred_img)
    axs[2].set_title("Prediction Overlay", pad=5)
    axs[2].axis('off')
    if title_text:
        fig.suptitle(title_text, y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)

def create_triple_plot_with_metrics(original_img, overlay_gt_img, overlay_pred_img,
                                    title_text=None, dice_scores=None, bce_losses=None, total_frames=None):
    """
    Create a triple plot (original, ground truth overlay, prediction overlay)
    with an additional roll-over metrics graph below it.

    The metrics graph displays two curves:
      - Left y-axis: Inverse Dice (1 - Dice) with range [0, 1] (ticks at 0, 0.5, 1).
      - Right y-axis: BCE Loss with range [0, 3] (ticks at 0, 1, 2, 3).

    The x-axis is fixed to the total number of frames (with ticks every 20 frames).

    Parameters:
        original_img (PIL.Image): The original image.
        overlay_gt_img (PIL.Image): The ground truth overlay image.
        overlay_pred_img (PIL.Image): The prediction overlay image.
        title_text (str): Text to display as a suptitle.
        dice_scores (list): List of dice scores (each between 0 and 1) up to the current frame.
        bce_losses (list): List of BCE losses up to the current frame.
        total_frames (int): The fixed total number of frames for the x-axis.

    Returns:
        PIL.Image: The composed figure as a PIL image.
    """
    total_frames = total_frames if total_frames is not None else len(dice_scores)
    num_processed = len(dice_scores)  # number of frames processed so far

    # Create the figure with two rows: one for the images, one for the metrics graph.
    fig = plt.figure(figsize=(20, 8), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3)

    # --- Top Row: Triple Plot of Images ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.imshow(original_img)
    ax1.set_title("Original", pad=5)
    ax1.axis('off')

    ax2.imshow(overlay_gt_img)
    ax2.set_title("Ground Truth Overlay", pad=5)
    ax2.axis('off')

    ax3.imshow(overlay_pred_img)
    ax3.set_title("Prediction Overlay", pad=5)
    ax3.axis('off')

    if title_text is not None:
        fig.suptitle(title_text, y=0.98, fontsize=16)

    # --- Bottom Row: Metrics Roll-Over Graph ---
    ax_dice = fig.add_subplot(gs[1, :])
    ax_dice.set_xlim(1, total_frames)
    xticks = list(range(20, total_frames + 1, 20))
    if not xticks or xticks[-1] != total_frames:
        xticks.append(total_frames)
    ax_dice.set_xticks(xticks)
    ax_dice.set_xlabel("Frame")

    # Plot inverse Dice on the left y-axis.
    x_vals = list(range(1, num_processed + 1))
    dice_error = [1 - d for d in dice_scores]
    line_dice, = ax_dice.plot(x_vals, dice_error, marker='o', linewidth=2, label='1 - Dice')
    ax_dice.set_ylim(0, 1)
    ax_dice.set_yticks([0, 0.5, 1])
    ax_dice.set_ylabel("1 - Dice (lower is better)")

    # Create a second y-axis for the BCE loss.
    ax_bce = ax_dice.twinx()
    line_bce, = ax_bce.plot(x_vals, bce_losses, marker='x', linewidth=2, color='tab:orange', label='BCE')
    ax_bce.set_ylim(0, 3)
    ax_bce.set_yticks([0, 1, 2, 3])
    ax_bce.set_ylabel("BCE Loss (lower is better)")

    # Optionally, add a legend combining both curves.
    lines = [line_dice, line_bce]
    labels = [line.get_label() for line in lines]
    ax_dice.legend(lines, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)

def create_gif(frames_list, output_path, fps=2):
    frames_np = [np.array(img) for img in frames_list]
    imageio.mimsave(output_path, frames_np, fps=fps)
    logging.info(f"Saved GIF to {output_path}")

def create_mp4(frames_list, output_path, fps=10):
    # Convert PIL images to NumPy arrays if needed
    frames_np = [np.array(frame) for frame in frames_list]
    clip = ImageSequenceClip(frames_np, fps=fps)
    clip.write_videofile(output_path, codec='libx264')

# --------------------------
# Processing Functions
# --------------------------
def process_single_image(img_path, net, device, args, mask_values, has_gt, imgs_dir, masks_dir):
    img = Image.open(img_path)
    masks = predict_img(net, img, device=device, thresholds=args.mask_thresholds)
    threshold = args.mask_thresholds[0]
    pred_mask = masks[threshold]
    overlay_pred = overlay_prediction_on_image(img, pred_mask, color=(255, 0, 255), alpha=0.3)

    # Get predicted probabilities (for BCE loss)
    img_conv = img.convert('L') if net.n_channels == 1 else img.convert('RGB')
    img_np = np.array(img_conv, dtype=np.float32)
    img_np = BasicDataset.preprocess(img_np, is_mask=False)
    if net.n_channels == 1 and img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=0)
    elif net.n_channels == 3 and img_np.ndim == 3:
        img_np = np.transpose(img_np, (2, 0, 1))
    img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img_torch).cpu()
        orig_size = (img.size[1], img.size[0])
        output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=False)
        probs = torch.sigmoid(output).squeeze()

    overlay_gt = None
    dice_score = None
    bce_loss_val = None
    if has_gt:
        gt_mask_path = masks_dir / f"{img_path.stem}_bolus.png"
        if gt_mask_path.is_file():
            gt_img = Image.open(gt_mask_path)
            gt_arr = np.array(gt_img, dtype=np.uint8)
            if gt_arr.ndim == 3:
                gt_arr = gt_arr[:, :, 0]
            gt_binary = (gt_arr > 128).astype(np.uint8)
            intersection = np.sum(pred_mask * gt_binary)
            dice_score = (2. * intersection) / (np.sum(pred_mask) + np.sum(gt_binary) + 1e-6)
            gt_tensor = torch.tensor(gt_binary, dtype=torch.float32)
            bce_loss_val = F.binary_cross_entropy(probs, gt_tensor, reduction='mean').item()
            overlay_gt = overlay_prediction_on_image(img, gt_arr, color=(0, 255, 0), alpha=0.3)
        else:
            logging.warning(f"Ground truth mask not found: {gt_mask_path}")

    # Optionally save raw mask
    if not args.no_save:
        raw_output_path = args.raw_output_dir / f"{img_path.stem}_mask.png"
        mask_to_image(pred_mask, mask_values).save(raw_output_path)

    return img, overlay_pred, overlay_gt, dice_score, bce_loss_val


def process_frames_group(frame_paths, group_name, net, device, args, mask_values, has_gt, masks_dir):
    """Process a group of frames (either from CSV mapping or all images together)."""
    frames_for_video = []
    dice_scores_list = []
    bce_losses_list = []

    total_frames = len(frame_paths)
    for idx, img_path in enumerate(frame_paths, start=1):
        img, overlay_pred, overlay_gt, dice_score, bce_loss = process_single_image(
            img_path, net, device, args, mask_values, has_gt, None, masks_dir)

        dice_scores_list.append(dice_score if dice_score is not None else 0)
        bce_losses_list.append(bce_loss if bce_loss is not None else 0)
        metrics_info = (f"1-Dice: {1 - dice_score:.2f} | BCE: {bce_loss:.2f}"
                        if dice_score is not None and bce_loss is not None else "")
        title_text = (f"{group_name} | Frame: {idx}/{total_frames} | Threshold: {args.mask_thresholds[0]} | "
                      f"FPS: {args.fps} | {metrics_info}")

        # Use triple plot with metrics if flag is enabled and ground truth is available
        if has_gt and overlay_gt is not None:
            if getattr(args, 'plot_metrics', False):
                plot_img = create_triple_plot_with_metrics(
                    img, overlay_gt, overlay_pred, title_text=title_text,
                    dice_scores=dice_scores_list, bce_losses=bce_losses_list, total_frames=total_frames
                )
            else:
                plot_img = create_triple_plot(img, overlay_gt, overlay_pred, title_text=title_text)
        else:
            plot_img = create_double_plot(img, overlay_pred, title_text=title_text)

        if args.save_frame_plots:
            frame_plot_path = args.viz_pred_dir / f"{group_name}_{img_path.stem}_plot.png"
            plot_img.save(frame_plot_path)
        frames_for_video.append(plot_img)
    return frames_for_video, dice_scores_list, bce_losses_list


def save_metrics(metrics_dict, output_dir, test_name):
    """Save metrics as a CSV and a TXT file."""
    # Save CSV if required
    if metrics_dict.get("save_csv", False):
        csv_path = output_dir / f"{test_name}_metrics.csv"
        df = pd.DataFrame(metrics_dict["data"])
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved metrics CSV to {csv_path}")
    # Save TXT file
    txt_path = output_dir / f"{test_name}_metrics.txt"
    with open(txt_path, "w") as f:
        for key, value in metrics_dict["summary"].items():
            f.write(f"{key}: {value}\n")
    logging.info(f"Saved metrics TXT to {txt_path}")

# --------------------------
# CLI and Main
# --------------------------
def get_args():
    parser = argparse.ArgumentParser(description='Predict segmentation masks using a UNet model')
    parser.add_argument('--test-name', type=str, required=True, help='Test name identifier.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save outputs.')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing "imgs" and optionally "masks".')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='CSV mapping file with columns for video_name and frame information.')
    parser.add_argument('--no-gt', action='store_true', help='If set, do not use ground truth.')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save raw output masks.')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize images as they are processed.')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for input images.')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling.')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes (1 for binary).')
    parser.add_argument('--channels', type=int, default=1, help='Number of input channels (default=1 for grayscale).')
    parser.add_argument('--mask-thresholds', '-t', type=float, nargs='+', default=[0.5],
                        help='List of probability thresholds (e.g., 0.3 0.5 0.7).')
    # Output type flags
    parser.add_argument('--save-frame-plots', action='store_true', help='Save per-image plots.')
    parser.add_argument('--save-video-gifs', action='store_true', help='Create GIF videos from frames.')
    parser.add_argument('--save-video-mp4s', action='store_true', help='Create MP4 videos from frames.')
    parser.add_argument('--save-metrics-csv', action='store_true', help='Save metrics as a CSV file.')
    parser.add_argument('--fps', type=int, default=7, help='Frames per second for GIF output.')
    # New flag: Use triple plot with metrics
    parser.add_argument('--plot-metrics', action='store_true', help='Use triple plot with metrics if ground truth is available.')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create output directories
    run_output_dir = Path(args.output_dir) / args.test_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_output_dir = run_output_dir / 'raw'
    args.raw_output_dir.mkdir(exist_ok=True)
    args.viz_pred_dir = run_output_dir / 'viz_pred'
    args.viz_pred_dir.mkdir(exist_ok=True)
    video_gif_dir = run_output_dir / 'video_gifs'
    video_gif_dir.mkdir(exist_ok=True)
    video_mp4_dir = run_output_dir / 'video_mp4s'
    video_mp4_dir.mkdir(exist_ok=True)

    # Save run info
    with open(run_output_dir / "run_info.txt", 'w') as f:
        f.write(f"Model used: {args.model_path}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Test name: {args.test_name}\n")
    logging.info(f"Run information written to {run_output_dir / 'run_info.txt'}")

    # Load model
    net, device, mask_values = load_model(Path(args.model_path), args.channels, args.classes, args.bilinear)

    # Directories for images and (if available) masks
    data_dir = Path(args.data_dir)
    imgs_dir = data_dir / 'imgs'
    masks_dir = data_dir / 'masks'
    if not imgs_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {imgs_dir}")
    has_gt = (not args.no_gt) and masks_dir.is_dir()

    # Load CSV if provided; otherwise process all images together
    if args.csv_path and Path(args.csv_path).is_file():
        logging.info(f"Reading CSV mapping: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        df = df[df["split"] == "test"]
        grouped = df.groupby("video_name")
    else:
        logging.info("No CSV mapping provided; processing all images together.")
        grouped = None

    # Global metrics collection for saving later.
    global_metrics = {"data": [], "summary": {}, "save_csv": args.save_metrics_csv}

    if grouped is None:
        # Process individual images and/or create one video from all frames if requested.
        all_imgs = sorted(list(imgs_dir.glob('*.png')))
        frames, dice_list, bce_list = process_frames_group(all_imgs, "all_frames", net, device, args,
                                                             mask_values, has_gt, masks_dir)
        # Optionally save video outputs from the collected frames
        if args.save_video_gifs and frames:
            gif_path = video_gif_dir / "all_frames.gif"
            create_gif(frames, gif_path, fps=args.fps)

        if args.save_video_mp4s and frames:
            mp4_path = video_mp4_dir / "all_frames.mp4"
            create_mp4(frames, str(mp4_path), fps=args.fps)
        # (MP4 saving would require additional video encoding libraries; here you can integrate one if needed.)
        # Record overall metrics
        if dice_list:
            global_metrics["summary"] = {
                "Overall Dice Mean": np.mean(dice_list),
                "Overall BCE Mean": np.mean(bce_list)
            }
            for img_path, d, b in zip(all_imgs, dice_list, bce_list):
                global_metrics["data"].append({"image": img_path.name, "dice": d, "bce": b})
    else:
        # Process each video group separately.
        for video_name, group in grouped:
            group = group.sort_values("new_frame_name")
            frame_paths = []
            for _, row in group.iterrows():
                # Assumes the CSV column "new_frame_name" contains the frame filename.
                frame_path = imgs_dir / row["new_frame_name"]
                if frame_path.is_file():
                    frame_paths.append(frame_path)
                else:
                    logging.warning(f"Image not found: {frame_path}")
            if not frame_paths:
                continue
            frames, dice_list, bce_list = process_frames_group(frame_paths, video_name.replace('.mp4', ''),
                                                                 net, device, args, mask_values,
                                                                 has_gt, masks_dir)
            if args.save_video_gifs and frames:
                gif_path = video_gif_dir / f"{video_name.replace('.mp4','')}.gif"
                create_gif(frames, gif_path, fps=args.fps)

            if args.save_video_mp4s and frames:
                mp4_path = video_mp4_dir / f"{video_name.replace('.mp4', '')}.mp4"
                create_mp4(frames, str(mp4_path), fps=args.fps)

            # MP4 output placeholder (integration with a video writer needed)
            # Record per-video metrics
            video_summary = {
                "Video": video_name,
                "Dice Mean": np.mean(dice_list) if dice_list else None,
                "BCE Mean": np.mean(bce_list) if bce_list else None
            }
            global_metrics["data"].append(video_summary)

    # Save overall metrics to a TXT file and optionally CSV.
    save_metrics(global_metrics, run_output_dir, args.test_name)
    logging.info("All predictions complete!")

if __name__ == '__main__':
    main()
