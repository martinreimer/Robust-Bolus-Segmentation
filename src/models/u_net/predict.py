"""
Predict segmentation masks using a pre-trained UNet model.

This script takes a super directory (with 'imgs/' and optionally 'masks/' subfolders)
and performs predictions on images using a UNet model. It generates visualizations as:
    - Double plots (Original image and Prediction Overlay) if ground truth is not provided.
    - Triple plots (Original, Ground Truth Overlay, and Prediction Overlay) if ground truth is available.
If a CSV overview file (with columns "video_name" and "frame_idx") is provided, the script groups
images by video and creates an animated GIF for each video. Each frame of the GIF will display a
suptitle containing the video name, current frame (e.g. "1/68"), the prediction threshold, and the FPS.

Example CLI commands:
    python .\predict.py --run-name run-20250308_211056 --model-file checkpoint_epoch14.pth --data-dir D:/Martin/thesis/data/processed/dataset_0228_final/test/
 --csv-path D:/Martin/thesis/data/processed/dataset_0228_final/data_overview.csv -v --gif-fps 5 -t [0.8, 0.2]
    python predict.py --run-name run-20250308_211056 --model-file checkpoint_epoch14.pth --data-dir "D:/Martin/thesis/data/processed/dataset_0228_final/test" --csv-path "data_overview.csv" --gif-fps 2
    python predict.py --run-name run-20250308_211056 --model-file checkpoint_epoch14.pth --data-dir "D:/Martin/thesis/data/processed/dataset_0228_final/test" --no-gt
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

# surpress torch future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

import pandas as pd  # For reading data_overview.csv
import imageio       # For creating GIFs

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from tqdm import tqdm


def find_runs_dir(run_name=None, runs_root='runs'):
    runs_root = Path(runs_root)
    if run_name:
        candidate = runs_root / run_name
        if candidate.is_dir():
            return candidate
        else:
            logging.warning(f"[find_runs_dir] Could not find run directory {candidate}, returning None.")
            return None
    else:
        run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith('run-')]
        if not run_dirs:
            logging.warning(f"[find_runs_dir] No run directories found in {runs_root}")
            return None
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return run_dirs[0]


def predict_img(net, full_img: Image.Image, device: torch.device, scale_factor=1.0, thresholds=[0.5]):
    """Predict masks for a single image using different thresholds."""
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
    """
    Convert the numeric mask to a PIL image based on `mask_values`.
    For binary segmentation with mask_values=[0,255], 0 maps to 0 and 1 maps to 255.
    """
    if len(mask_values) == 2 and set(mask_values) == {0, 1}:
        out = np.zeros(mask.shape, dtype=np.uint8)
        out[mask == 1] = 255
        return Image.fromarray(out)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for i, val in enumerate(mask_values):
        out[mask == i] = val
    return Image.fromarray(out)


def overlay_prediction_on_image(img_pil: Image.Image, mask: np.ndarray, color=(255, 0, 255), alpha=0.3):
    """
    Overlay the prediction mask on the original image using a fixed color and transparency.
    """
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_np = np.array(img_pil)
    overlay = np.zeros_like(img_np)
    overlay[mask != 0] = color
    result = (alpha * overlay + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(result)


def create_double_plot(original_img, overlay_pred_img, title_text=None):
    """
    Create a double plot (side-by-side) with:
      - Left: Original image
      - Right: Prediction overlay.
    If title_text is provided, it is added as a suptitle.
    The figure is created with increased size (16x8 inches) and high DPI (300) and minimal spacing between subplots.
    Returns the composed figure as a PIL Image.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    plt.subplots_adjust(wspace=0, hspace=0)
    axs[0].imshow(original_img)
    axs[0].set_title("Original", pad=5)
    axs[0].axis('off')
    axs[1].imshow(overlay_pred_img)
    axs[1].set_title("Prediction Overlay", pad=5)
    axs[1].axis('off')
    if title_text is not None:
        fig.suptitle(title_text, y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)


def create_triple_plot(original_img, overlay_gt_img, overlay_pred_img, title_text=None):
    """
    Create a triple plot with:
      - Left: Original image
      - Middle: Ground truth overlay.
      - Right: Prediction overlay.
    If title_text is provided, it is added as a suptitle.
    The figure is created with increased size (24x8 inches) and high DPI (300) and minimal spacing between subplots.
    Returns the composed figure as a PIL Image.
    """
    fig, axs = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    plt.subplots_adjust(wspace=0, hspace=0)
    axs[0].imshow(original_img)
    axs[0].set_title("Original", pad=5)
    axs[0].axis('off')
    axs[1].imshow(overlay_gt_img)
    axs[1].set_title("Ground Truth Overlay", pad=5)
    axs[1].axis('off')
    axs[2].imshow(overlay_pred_img)
    axs[2].set_title("Prediction Overlay", pad=5)
    axs[2].axis('off')
    if title_text is not None:
        fig.suptitle(title_text, y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)


def create_gif(frames_list, output_path, fps=2):
    """
    Create a GIF from a list of PIL image frames.
    """
    frames_np = [np.array(img) for img in frames_list]
    imageio.mimsave(output_path, frames_np, fps=fps)
    logging.info(f"Saved GIF to {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name of a specific run directory in "runs/". If not specified, the newest run is used.')
    parser.add_argument('--runs-root', type=str, default='runs',
                        help='Parent directory containing run subdirs (default="runs").')
    parser.add_argument('--model-file', type=str, default='model.pth',
                        help='Name of the model file in the run directory (default="model.pth").')
    parser.add_argument('--data-dir', type=str,
                        default='../../../data/processed/dataset_first_experiments/test',
                        help='Super directory containing "imgs" and optionally "masks" subfolders.')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help='Do not save the raw output masks.')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed.')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images.')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling.')
    parser.add_argument('--classes', '-c', type=int, default=1,
                        help='Number of classes (1 for binary, else >1).')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of input channels (default=1 for grayscale).')
    parser.add_argument('--mask-thresholds', '-t', type=float, nargs='+', default=[0.5],
                        help='List of probability thresholds to apply (e.g., 0.3 0.5 0.7).')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to a CSV that associates video_name and frame_idx. If provided, GIFs are created per video.')
    parser.add_argument('--no-gt', action='store_true',
                        help='If set, do not load ground truth. Generate double plots instead of triple plots.')
    parser.add_argument('--gif-fps', type=int, default=2,
                        help='Frames per second for output GIFs.')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Determine the run directory
    run_dir = find_runs_dir(run_name=args.run_name, runs_root=args.runs_root)
    if run_dir is None:
        raise FileNotFoundError("[main] No valid run directory found. "
                                "Either specify --run-name or ensure runs/ is not empty.")

    # Construct path to the model file
    model_path = run_dir / "checkpoints" / args.model_file
    if not model_path.is_file():
        raise FileNotFoundError(f"[main] Model file {model_path} does not exist!")

    # Create timestamped output folder and subfolders
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path('output') / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = run_output_dir / 'raw'
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    viz_pred_dir = run_output_dir / 'viz_pred'
    viz_pred_dir.mkdir(parents=True, exist_ok=True)
    video_gif_dir = run_output_dir / 'video_gifs'
    video_gif_dir.mkdir(parents=True, exist_ok=True)
    info_file = run_output_dir / "run_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Model used: {model_path}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Date/Time: {timestamp}\n")
    logging.info(f"Wrote run information to: {info_file}")

    # Build the UNet
    logging.info("[main] Building UNet...")
    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[main] Using device: {device}")
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info("[main] Model loaded successfully!")

    # Determine where images and masks are located
    data_dir = Path(args.data_dir)
    imgs_dir = data_dir / 'imgs'
    masks_dir = data_dir / 'masks'  # Only used if ground truth is enabled
    if not imgs_dir.is_dir():
        raise FileNotFoundError(f"[main] Could not find images directory: {imgs_dir}")
    has_gt = (not args.no_gt) and masks_dir.is_dir()

    # If CSV is provided, group images by video; else process images individually
    if args.csv_path and Path(args.csv_path).is_file():
        logging.info(f"[main] Reading data overview CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        df = df[df["split"] == "test"]
        grouped = df.groupby("video_name")
    else:
        logging.info("[main] No CSV provided or file not found; processing images individually.")
        grouped = None

    # For images/masks, we assume filenames are just the frame index (e.g., "1181.png")
    # and masks have a "_bolus" suffix (e.g., "1181_bolus.png").
    def get_image_path(video_name, frame_idx):
        return imgs_dir / str(frame_idx)

    def get_mask_path(video_name, frame_idx):
        frame_idx = str(str(frame_idx)).replace('.png', '_bolus.png"')
        return masks_dir / frame_idx

    def process_single_image(img_path):
        img = Image.open(img_path)
        masks = predict_img(net, img, device=device, thresholds=args.mask_thresholds)
        threshold = args.mask_thresholds[0]
        pred_mask = masks[threshold]
        overlay_pred = overlay_prediction_on_image(img_pil=img, mask=pred_mask, color=(255, 0, 255), alpha=0.3)
        overlay_gt = None
        if has_gt:
            mask_path = masks_dir / f"{img_path.stem}_bolus.png"
            if mask_path.is_file():
                gt_img_pil = Image.open(mask_path)
                gt_arr = np.array(gt_img_pil, dtype=np.uint8)
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr[:, :, 0]
                overlay_gt = overlay_prediction_on_image(img_pil=img, mask=gt_arr, color=(0, 255, 0), alpha=0.3)
            else:
                logging.warning(f"Ground truth mask not found: {mask_path}")
        if not args.no_save:
            raw_mask_path = raw_output_dir / f"{img_path.stem}_mask.png"
            mask_to_image(pred_mask, mask_values).save(raw_mask_path)
        return img, overlay_pred, overlay_gt

    if grouped is None:
        # Process images individually (non-video GIF scenario)
        all_imgs = sorted(list(imgs_dir.glob('*.png')))
        for img_path in tqdm(all_imgs, desc="Predicting on images"):
            original_img, overlay_pred, overlay_gt = process_single_image(img_path)
            if has_gt and overlay_gt is not None:
                triple_img = create_triple_plot(original_img, overlay_gt, overlay_pred)
                triple_img.save(viz_pred_dir / f"{img_path.stem}_triple.png")
            else:
                double_img = create_double_plot(original_img, overlay_pred)
                double_img.save(viz_pred_dir / f"{img_path.stem}_double.png")
    else:
        for video_name, group in grouped:
            logging.info(f"Processing video: {video_name} with {len(group)} frames.")
            group = group.sort_values("new_frame_name")
            frames_for_gif = []
            total_frames = len(group)
            video_info = f"Video: {video_name.replace('.mp4', '')}"
            threshold_info = f"Threshold: {args.mask_thresholds[0]}"
            fps_info = f"FPS: {args.gif_fps}"
            print(f"Processing video: {video_name} with {total_frames} frames.")
            for idx, (_, row) in enumerate(group.iterrows(), start=1):
                frame_idx = row["new_frame_name"]
                img_path = get_image_path(video_name, frame_idx)

                if not img_path.is_file():
                    logging.warning(f"Image not found: {img_path}")
                    continue

                original_img, overlay_pred, overlay_gt = process_single_image(img_path)

                # Build a suptitle for the figure
                title_text = f"{video_info} | Frame: {idx}/{total_frames} | {threshold_info} | {fps_info}"

                if has_gt and overlay_gt is not None:
                    frame_plot = create_triple_plot(
                        original_img, overlay_gt, overlay_pred, title_text=title_text
                    )
                    # Example filename for triple plot
                    out_plot_path = viz_pred_dir / f"{video_name.replace('.mp4', '')}_{frame_idx}_triple.png"
                else:
                    frame_plot = create_double_plot(
                        original_img, overlay_pred, title_text=title_text
                    )
                    # Example filename for double plot
                    out_plot_path = viz_pred_dir / f"{video_name.replace('.mp4', '')}_{frame_idx}_double.png"

                # 1) Save the individual frame plot
                frame_plot.save(out_plot_path)

                # 2) Append to the GIF frames
                frames_for_gif.append(frame_plot)
            # After processing all frames in this video, build the GIF
            if frames_for_gif:
                gif_path = video_gif_dir / f"{video_name.replace('.mp4', '')}.gif"
                create_gif(frames_for_gif, gif_path, fps=args.gif_fps)

    logging.info("[main] All predictions complete!")


if __name__ == '__main__':
    main()
