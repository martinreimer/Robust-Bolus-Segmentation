r'''
Example CLI:
    python predict.py \
        --test-name winter-valley-234 \
        --model-path /path/to/checkpoint_epoch11.pth \
        --output-dir /path/to/output \
        --data-dir /path/to/data/processed/val/ \
        --csv-path /path/to/data_overview.csv \
        -v \
        -t 0.8 \
        --save-metrics-csv \
        --save-video-mp4s \
        --fps 10 \
        --dataset-split val \
        --plot-metrics

python predict.py --test-name dandy-energy-320 --model-path D:\Martin\thesis\training_runs\U-Net\runs\dandy-energy-320\checkpoints\checkpoint_epoch36.pth --output-dir D:\Martin\thesis\test_runs --data-dir D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop/val --csv-path D:/Martin/thesis/data/processed/dataset_labelbox_export_test_2504_test_final_roi_crop/data_overview.csv --save-metrics-csv --save-video-mp4s --fps 10 --dataset-split val --plot-metrics

Description:
    This script loads a pre-trained U-Net model from segmentation-models-pytorch (SMP),
    performs segmentation on grayscale images, and produces:
      - Raw binary masks (PNG)
      - Overlay visualizations per frame
      - GIF and MP4 videos
      - Dice & BCE metrics (CSV and TXT)

Features:
    - Single-channel (grayscale) input only
    - Optional ground-truth (--no-gt)
    - CSV-driven frame grouping for videos (--csv-path)
    - Frame-by-frame or grouped processing
    - Metrics plotting (--plot-metrics)

Requirements:
    - segmentation-models-pytorch
    - torch, torchvision
    - moviepy, imageio
    - matplotlib, pandas, PIL

'''
import argparse
import logging
import os
from pathlib import Path
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import pandas as pd
import imageio
from tqdm import tqdm

import segmentation_models_pytorch as smp
from .utils.data_loading import BasicDataset
from .utils.utils import plot_img_and_mask  # if needed
from moviepy.editor import ImageSequenceClip
from .unet import UNet
# --------------------------
# Helper Functions
# --------------------------

import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import binary_erosion
from skimage.segmentation import find_boundaries

def compute_surface_distances(pred_mask, gt_mask):
    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        # Both masks empty: perfect alignment
        return np.array([0.0]), 0.0, 0.0

    pred_surface = find_boundaries(pred_mask, mode='outer')
    gt_surface = find_boundaries(gt_mask, mode='outer')

    pred_points = np.argwhere(pred_surface)
    gt_points = np.argwhere(gt_surface)
    '''
    if len(pred_points) == 0:
        logging.warning("Prediction has no boundaries.")
    if len(gt_points) == 0:
        logging.warning("Ground truth has no boundaries.")
    '''
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.array([]), np.nan, np.nan

    dists_pred_to_gt = cdist(pred_points, gt_points).min(axis=1)
    dists_gt_to_pred = cdist(gt_points, pred_points).min(axis=1)

    surface_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    return surface_dists, dists_pred_to_gt.mean(), dists_gt_to_pred.mean()


def compute_hd95_asd(pred_mask, gt_mask):
    surface_dists, mean_pred_to_gt, mean_gt_to_pred = compute_surface_distances(pred_mask, gt_mask)
    if surface_dists is None or surface_dists.size == 0:
        return None, None

    hd95 = np.percentile(surface_dists, 95)
    asd = (mean_pred_to_gt + mean_gt_to_pred) / 2
    return hd95, asd


def load_old_model(model_path, channels, classes, bilinear):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    net = None#UNet(**config)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device=device)
    mask_values = checkpoint.pop('mask_values', [0, 1])
    logging.info("Model loaded successfully!")
    return net, device, mask_values

def load_model(model_path: Path):
    """
    Load a segmentation-models-pytorch U-Net model from checkpoint.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    print(f"MODEL SOURCE: " + config["model_source"])
    if config["model_source"] == "smp":
        # print config keys
        smp_model = config["smp_model"]
        print(f"MODEL SOURCE: {smp_model}")
        # remove model_source key from config
        config.pop("model_source", None)
        config.pop("smp_model", None)
        config.pop("decoder_interpolation", None)
        config.pop("decoder_use_norm", None)
        if smp_model.lower() == "unet":
            net = smp.Unet(**config)
        elif smp_model.lower() == "unetplusplus":
            net = smp.UnetPlusPlus(**config)
        elif smp_model.lower() == "segformer":
            net = smp.Segformer(**config)
        else:
            raise ValueError(f"Unknown SMP model: {smp_model}")

    elif config["model_source"] == "custom":
        config.pop("model_source", None)
        net = UNet(**config)
    else:
        # throw an error
        raise ValueError(f"Unknown model source: {config['model_source']}")

    net.load_state_dict(checkpoint['state_dict'])
    net.to(device=device)
    # Optional mask value mapping
    mask_values = checkpoint.get('mask_values', [0, 1])
    logging.info("Model loaded successfully on %s", device)
    return net, device, mask_values


def predict_img(net: torch.nn.Module,
                full_img: Image.Image,
                device: torch.device,
                scale_factor: float = 1.0,
                thresholds: list = [0.5]):
    """
    Run inference on a single PIL image, return binary masks for given thresholds.
    """
    net.eval()
    # Always grayscale input
    img_gray = full_img.convert('L')
    img_np = np.array(img_gray, dtype=np.float32)
    img_np = BasicDataset.preprocess(img_np, is_mask=False)

    # Ensure channel-first
    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=0)

    img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img_torch)  # shape: [1, classes, H, W]
        output = output.cpu()
        # Resize to original size
        orig_size = (full_img.size[1], full_img.size[0])
        output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=False)
        probs = torch.sigmoid(output)

        masks = {th: (probs > th).long().squeeze().numpy() for th in thresholds}
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


def overlay_prediction_on_image(img_pil: Image.Image,
                                mask: np.ndarray,
                                color=(255, 0, 255),
                                alpha=0.3):
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_np = np.array(img_pil)
    overlay = np.zeros_like(img_np)
    overlay[mask != 0] = color
    result = (alpha * overlay + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(result)


def create_double_plot(original_img, overlay_pred_img, title_text=None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    axs[0].imshow(original_img); axs[0].set_title("Original"); axs[0].axis('off')
    axs[1].imshow(overlay_pred_img); axs[1].set_title("Prediction Overlay"); axs[1].axis('off')
    if title_text:
        fig.suptitle(title_text, y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)  # RGBA
    buf = buf[:, :, :3]  # drop alpha to get RGB
    plt.close(fig)
    return Image.fromarray(buf)


def create_triple_plot(original_img, overlay_gt_img, overlay_pred_img, title_text=None):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), dpi=100)
    axs[0].imshow(original_img); axs[0].set_title("Original"); axs[0].axis('off')
    axs[1].imshow(overlay_gt_img); axs[1].set_title("Ground Truth Overlay"); axs[1].axis('off')
    axs[2].imshow(overlay_pred_img); axs[2].set_title("Prediction Overlay"); axs[2].axis('off')
    if title_text:
        fig.suptitle(title_text, y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)  # RGBA
    buf = buf[:, :, :3]  # drop alpha to get RGB
    plt.close(fig)
    return Image.fromarray(buf)

def create_triple_plot_with_metrics(original_img, overlay_gt_img, overlay_pred_img,
                                    title_text=None, dice_scores=None, total_frames=None):
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
    line_dice, = ax_dice.plot(x_vals, dice_scores, marker='o', linewidth=2, label='Dice')
    ax_dice.set_ylim(0, 1)
    ax_dice.set_yticks([0, 0.5, 1])
    ax_dice.set_ylabel("Dice (higher is better)")

    # Optionally, add a legend combining both curves.
    lines = [line_dice]
    labels = [line.get_label() for line in lines]
    ax_dice.legend(lines, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # draw and convert to image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)  # RGBA
    buf = buf[:, :, :3]  # drop alpha to get RGB
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
def process_single_image(img_path, net, device, args, mask_values, has_gt, masks_dir):
    """
    Runs inference on one frame, computes Dice and confusion (TP/TN/FP/FN).
    Returns:
        img (PIL), overlay_pred (PIL), overlay_gt (PIL or None),
        dice_score (float or None), tp (int), tn (int), fp (int), fn (int)
    """
    img = Image.open(img_path)
    masks = predict_img(net, img, device=device, thresholds=args.mask_thresholds)
    threshold = args.mask_thresholds[0]
    pred_mask = masks[threshold]
    overlay_pred = overlay_prediction_on_image(img, pred_mask, color=(255, 0, 255), alpha=0.3)

    dice_score = None
    tp = tn = fp = fn = 0
    overlay_gt = None
    hd95_value = asd_value = None

    if has_gt:
        gt_mask_path = masks_dir / f"{img_path.stem}_bolus.png"
        if gt_mask_path.is_file():
            gt_img = Image.open(gt_mask_path)
            gt_arr = np.array(gt_img.convert('L'), dtype=np.uint8)
            gt_binary = (gt_arr > 128).astype(np.uint8)

            # Dice
            intersection = np.sum(pred_mask * gt_binary)
            if pred_mask.sum() == 0 and gt_binary.sum() == 0:
                dice_score = 1.0
            else:
                dice_score = (2. * intersection) / (pred_mask.sum() + gt_binary.sum() + 1e-6)

            # Confusion counts
            pred_bool = pred_mask.astype(bool)
            gt_bool   = gt_binary.astype(bool)
            tp = int(np.logical_and(pred_bool, gt_bool).sum())
            tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
            fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
            fn = int(np.logical_and(~pred_bool, gt_bool).sum())

            overlay_gt = overlay_prediction_on_image(img, gt_arr, color=(0, 255, 0), alpha=0.3)
            # HD95 and ASD
            try:
                hd95_value, asd_value = compute_hd95_asd(pred_mask, gt_binary)
            except Exception as e:
                logging.warning(f"Could not compute HD95/ASD for {img_path.name}: {e}")
                hd95_value = asd_value = None

        else:
            logging.warning(f"GT mask not found: {gt_mask_path}")

    # optionally save raw mask
    if not args.no_save:
        out_path = args.raw_output_dir / f"{img_path.stem}_mask.png"
        mask_to_image(pred_mask, mask_values).save(out_path)

    return img, overlay_pred, overlay_gt, dice_score, tp, tn, fp, fn, hd95_value, asd_value

def process_frames_group(frame_paths, group_name, net, device, args, mask_values, has_gt, masks_dir):
    """
    Processes all frames of one video/group, accumulating:
      - frames_for_video (PIL list)
      - dice_scores_list (list of floats)
      - tp_list, tn_list, fp_list, fn_list (lists of ints)
    """
    frames_for_video = []
    dice_scores_list = []
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    hd95_list = []
    asd_list = []
    total_frames = len(frame_paths)

    for idx, img_path in enumerate(frame_paths, start=1):
        img, overlay_pred, overlay_gt, dice, tp, tn, fp, fn, hd95_value, asd_value  = process_single_image(
            img_path, net, device, args, mask_values, has_gt, masks_dir)

        dice_scores_list.append(dice if dice is not None else 0)
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        hd95_list.append(hd95_value if hd95_value is not None else np.nan)
        asd_list.append(asd_value if asd_value is not None else np.nan)

        title = f"{group_name} | Frame: {idx}/{total_frames} | Dice: {dice:.2f}"
        if has_gt and overlay_gt is not None and args.plot_metrics:
            plot_img = create_triple_plot_with_metrics(
                img, overlay_gt, overlay_pred,
                title_text=title,
                dice_scores=dice_scores_list,
                total_frames=total_frames
            )
        else:
            plot_img = create_double_plot(img, overlay_pred, title_text=title)

        if args.save_frame_plots:
            plot_path = args.viz_pred_dir / f"{group_name}_{img_path.stem}_plot.png"
            plot_img.save(plot_path)

        frames_for_video.append(plot_img)

    return frames_for_video, dice_scores_list, tp_list, tn_list, fp_list, fn_list, hd95_list, asd_list


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
    parser.add_argument('--dataset-split', type=str, default=None, help='Parse Train / Val / Test Videos from Data_Overview.csv')
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

def calculate_metrics(dice_list, tp_list, tn_list, fp_list, fn_list, hd95_list, asd_list):
    results = {}
    # Record per-video metrics
    results['dice_mean'] = np.mean(dice_list)
    results['dice_median'] = np.median(dice_list)
    results['dice_25'] = np.percentile(dice_list, 25)
    results['dice_75'] = np.percentile(dice_list, 75)

    results['hd95'] = np.nanmean(hd95_list)
    results['asd'] = np.nanmean(asd_list)

    results['tp'] = sum(tp_list)
    results['tn'] = sum(tn_list)
    results['fp'] = sum(fp_list)
    results['fn'] = sum(fn_list)

    results['specificity'] = results['tn'] / (results['tn'] + results['fp'] + 1e-6)
    results['recall'] = results['tp'] / (results['tp'] + results['fn'] + 1e-6)
    results['precision'] = results['tp'] / (results['tp'] + results['fp'] + 1e-6)
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'] + 1e-6)
    results['iou'] = results['tp'] / (results['tp'] + results['fp'] + results['fn'] + 1e-6)
    # round to 4 decimal places all values all values in for loop
    for key in results.keys():
        if key in ['tp', 'tn', 'fp', 'fn']:
            results[key] = int(results[key])
        else:
            results[key] = round(results[key], 4)
    return results

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create base run directory (independent of split)
    base_dir = Path(args.output_dir) / f"{args.test_name}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # If user specified a split, make a subfolder for it; otherwise use base_dir itself
    if args.dataset_split in ('train', 'val', 'test'):
        run_output_dir = base_dir / args.dataset_split
    else:
        run_output_dir = base_dir
    run_output_dir.mkdir(exist_ok=True)

    # now sub‐dirs under run_output_dir
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
    net, device, mask_values = load_model(Path(args.model_path))

    # Directories for images and (if available) masks
    data_dir = Path(args.data_dir)
    imgs_dir = data_dir / 'imgs'
    masks_dir = data_dir / 'masks'
    if not imgs_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {imgs_dir}")
    has_gt = (not args.no_gt) and masks_dir.is_dir()

    # Load CSV if provided; otherwise process all images together
    if args.csv_path and Path(args.csv_path).is_file():
        if args.dataset_split not in ['train', 'val', 'test']:
            raise ValueError("dataset_split must be one of: train, val, test")
        logging.info(f"Reading CSV mapping: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        df = df[df["split"] == args.dataset_split]
        grouped = df.groupby("video_name")
    else:
        logging.info("No CSV mapping provided; processing all images together.")
        grouped = None

    # Global metrics collection for saving later.
    global_metrics = {"data": [], "summary": {}, "save_csv": args.save_metrics_csv}

    if grouped is None:
        # Process individual images and/or create one video from all frames if requested.
        all_imgs = sorted(list(imgs_dir.glob('*.png')))
        frames, dice_list, tp_list, tn_list, fp_list, fn_list, hd95_list, asd_list = process_frames_group(
            all_imgs, "all_frames", net, device, args, mask_values, has_gt, masks_dir)

        if args.save_video_gifs and frames:
            gif_path = video_gif_dir / "all_frames.gif"
            create_gif(frames, gif_path, fps=args.fps)

        if args.save_video_mp4s and frames:
            mp4_path = video_mp4_dir / "all_frames.mp4"
            create_mp4(frames, str(mp4_path), fps=args.fps)

        # Record overall metrics
        if dice_list:
            global_metrics["summary"] = {
                "Overall Dice Mean": np.mean(dice_list)
            }
            for img_path, d in zip(all_imgs, dice_list):
                global_metrics["data"].append({"image": img_path.name, "dice": d})
    else:
        dice_overall, frames_overall, tp_overall, tn_overall, fp_overall, fn_overall, hd95_overall, asd_overall = [], [], [], [], [], [], [], []
        # Process each video group separately.
        for video_name, group in grouped:
            group = group.sort_values("new_frame_name")
            frame_paths = []
            for _, row in group.iterrows():
                frame_path = imgs_dir / row["new_frame_name"]
                if frame_path.is_file():
                    frame_paths.append(frame_path)
                else:
                    logging.warning(f"Image not found: {frame_path}")
            if not frame_paths:
                continue

            frames, dice_list, tp_list, tn_list, fp_list, fn_list, hd95_list, asd_list = process_frames_group(
                frame_paths, video_name.replace('.mp4', ''), net, device, args, mask_values, has_gt, masks_dir)

            # Append to global metrics
            dice_overall.extend(dice_list)
            frames_overall.extend(frames)
            tp_overall.extend(tp_list)
            tn_overall.extend(tn_list)
            fp_overall.extend(fp_list)
            fn_overall.extend(fn_list)
            hd95_overall.extend(hd95_list)
            asd_overall.extend(asd_list)


            # Save video frames as GIF and MP4
            if args.save_video_gifs and frames:
                gif_path = video_gif_dir / f"{video_name.replace('.mp4','')}.gif"
                create_gif(frames, gif_path, fps=args.fps)

            if args.save_video_mp4s and frames:
                mp4_path = video_mp4_dir / f"{video_name.replace('.mp4', '')}.mp4"
                create_mp4(frames, str(mp4_path), fps=args.fps)

            # Record per-video metrics
            results = calculate_metrics(dice_list, tp_list, tn_list, fp_list, fn_list, hd95_list, asd_list)

            video_summary = {
                "Video": video_name,
                "Frames": len(frame_paths),
                "Dice Mean": results['dice_mean'],
                "Dice Median": results['dice_median'],
                "Dice 25th Percentile": results['dice_25'],
                "Dice 75th Percentile": results['dice_75'],
                "True Positives": results['tp'],
                "True Negatives": results['tn'],
                "False Positives": results['fp'],
                "False Negatives": results['fn'],
                "specificity": results['specificity'],
                "recall": results['recall'],
                "precision": results['precision'],
                "f1": results['f1'],
                "iou": results['iou'],
                "hd95": results['hd95'],
                "asd": results['asd'],
            }
            global_metrics["data"].append(video_summary)
            # --------------------------
            # Add Frame Count, AVG and Total Rows
            # --------------------------
        if global_metrics["data"]:
            df = pd.DataFrame(global_metrics["data"])
            # Total row
            results = calculate_metrics(dice_overall, tp_overall, tn_overall, fp_overall, fn_overall, hd95_overall, asd_overall)

            df_total = pd.Series({
                'Video': 'Total Frames',
                "Frames": len(frames_overall),
                "Dice Mean": results['dice_mean'],
                "Dice Median": results['dice_median'],
                "Dice 25th Percentile": results['dice_25'],
                "Dice 75th Percentile": results['dice_75'],
                "True Positives": results['tp'],
                "True Negatives": results['tn'],
                "False Positives": results['fp'],
                "False Negatives": results['fn'],
                "specificity": results['specificity'],
                "recall": results['recall'],
                "precision": results['precision'],
                "f1": results['f1'],
                "iou": results['iou'],
                "hd95": results['hd95'],
                "asd": results['asd'],
            })
            # Avg over videos row
            # Compute mean over videos (excluding Total row)
            metric_cols = ['Dice Mean', 'Dice Median', 'Dice 25th Percentile',
                           'Dice 75th Percentile', 'True Positives', 'True Negatives',
                           'False Positives', 'False Negatives', 'specificity',
                           'recall', 'precision', 'f1', 'iou', 'Frames', 'hd95', 'asd']
            df_avg = df[metric_cols].agg(np.nanmean).round(4)
            df_avg['Video'] = 'AVG Video'

            # Reorder and append rows
            df = pd.concat([df, df_avg.to_frame().T, df_total.to_frame().T], ignore_index=True)
            global_metrics["data"] = df.to_dict(orient='records')

    # Save overall metrics to a TXT file and optionally CSV.
    save_metrics(global_metrics, run_output_dir, args.test_name)
    logging.info("All predictions complete!")


if __name__ == '__main__':
    main()
