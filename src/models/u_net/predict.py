import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from tqdm import tqdm

def find_runs_dir(run_name=None, runs_root='runs'):
    runs_root = Path(runs_root)

    if run_name:
        # User explicitly provided a run name
        candidate = runs_root / run_name
        if candidate.is_dir():
            return candidate
        else:
            logging.warning(f"[find_runs_dir] Could not find run directory {candidate}, returning None.")
            return None
    else:
        # Pick the newest directory that starts with 'run-'
        run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith('run-')]
        if not run_dirs:
            logging.warning(f"[find_runs_dir] No run directories found in {runs_root}")
            return None

        # Sort by modification time descending
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return run_dirs[0]  # newest


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

        # Process each threshold separately
        masks = {}
        for threshold in thresholds:
            masks[threshold] = (torch.sigmoid(output) > threshold).long().squeeze().numpy()

    return masks  # Dictionary {threshold: mask_array}



def mask_to_image(mask: np.ndarray, mask_values):
    """
    Convert the numeric mask to a PIL image based on `mask_values`.
    For example, if mask_values=[0,255], we map 0->0, 1->255 for binary.
    If multi-class with e.g. mask_values=[0,127,255], we map class 0->0, class 1->127, class 2->255, etc.
    """
    if len(mask_values) == 2 and set(mask_values) == {0, 1}:
        # interpret as boolean => map 0->0, 1->255
        out = np.zeros(mask.shape, dtype=np.uint8)
        out[mask == 1] = 255
        return Image.fromarray(out)

    # Otherwise, create an 8-bit array
    out = np.zeros(mask.shape, dtype=np.uint8)
    for i, val in enumerate(mask_values):
        out[mask == i] = val
    return Image.fromarray(out)


def overlay_prediction_on_image(img_pil: Image.Image,
                                mask: np.ndarray,
                                color=(255, 0, 255),
                                alpha=0.3):
    """
    Create a visualization by overlaying the binary or multi-class mask on the original image in a single color.
    Now with alpha=0.3 for more transparency.
    """
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    img_np = np.array(img_pil)
    overlay = np.zeros_like(img_np)
    overlay[mask != 0] = color

    result = (alpha * overlay + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(result)


def overlay_pred_and_gt_on_image(img_pil: Image.Image,
                                 pred_mask: np.ndarray,
                                 gt_mask: np.ndarray,
                                 pred_color=(255, 0, 255),  # Purple
                                 gt_color=(0, 255, 0),     # Green
                                 alpha=0.3):
    """
    Overlay predicted mask in one color, and ground truth in another color, on the original image.
    """
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    base_np = np.array(img_pil)

    pred_overlay = np.zeros_like(base_np)
    pred_overlay[pred_mask != 0] = pred_color

    gt_overlay = np.zeros_like(base_np)
    gt_overlay[gt_mask != 0] = gt_color

    combined_overlay = pred_overlay + gt_overlay
    result = (alpha * combined_overlay + (1 - alpha) * base_np).astype(np.uint8)
    return Image.fromarray(result)


def create_triplet_plot(original_img,
                        overlay_gt_img,
                        overlay_pred_img,
                        save_path):
    """
    Creates a single figure with three subplots:
      - Left: Original image
      - Middle: Original image with ground truth overlay
      - Right: Original image with prediction overlay
    Saves to save_path.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Show the original image
    axs[0].imshow(original_img)
    axs[0].set_title("Original")
    axs[0].axis('off')

    # Show the ground truth overlay
    axs[1].imshow(overlay_gt_img)
    axs[1].set_title("Ground Truth Overlay")
    axs[1].axis('off')

    # Show the prediction overlay
    axs[2].imshow(overlay_pred_img)
    axs[2].set_title("Prediction Overlay")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name of a specific run directory in "runs/". If not specified, use the newest run.')
    parser.add_argument('--runs-root', type=str, default='runs',
                        help='Parent directory containing run subdirs, default="runs"')
    parser.add_argument('--model-file', type=str, default='model.pth',
                        help='Name of the model file in the run directory, default="model.pth"')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='Filenames of input images. Default is all images in ./data/test/imgs',
                        default=None)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Filenames of output images. If not given, automatically generated')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images.')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes (1 for binary, else >1)')
    parser.add_argument('--channels', type=int, default=1, help='Number of input channels (default=1 for grayscale)')
    parser.add_argument('--test-dir', type=str,
                        default='../../../data/processed/dataset_first_experiments/test/imgs',
                        help='Path to the folder containing test images (default to your dataset path).')
    parser.add_argument('--gt-dir', type=str,
                        default='../../../data/processed/dataset_first_experiments/test/masks',
                        help='Path to the folder containing ground truth masks (for optional visualization).')
    parser.add_argument('--mask-thresholds', '-t', type=float, nargs='+', default=[0.5],
                        help='List of probability thresholds to apply (e.g., 0.3 0.5 0.7).')

    return parser.parse_args()


def get_output_filenames(args, in_files, raw_output_dir):
    if args.output is not None:
        return args.output
    else:
        out_files = []
        for fn in in_files:
            stem = Path(fn).stem
            out_files.append(str(Path(raw_output_dir) / f"{stem}_OUT.png"))
        return out_files


def find_test_images(test_dir):
    test_dir_path = Path(test_dir)
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = []
    for ext in valid_exts:
        files.extend(test_dir_path.glob(f'*{ext}'))
    files = sorted(str(f) for f in files)
    return files


def get_gt_filename(img_filename, gt_dir, suffix='_bolus'):
    """
    If your test image is named "0.png", the GT is "0_bolus.png" in the same folder (gt_dir).
    """
    image_path = Path(img_filename)
    stem = image_path.stem  # e.g. "0"
    extension = image_path.suffix  # e.g. ".png"

    # Build the ground truth file name with the suffix
    candidate = Path(gt_dir) / f"{stem}{suffix}{extension}"
    if candidate.is_file():
        return str(candidate)
    else:
        return None


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Determine the run directory (from which we load the model)
    run_dir = find_runs_dir(run_name=args.run_name, runs_root=args.runs_root)
    if run_dir is None:
        raise FileNotFoundError("[main] No valid run directory found. "
                                "Either specify --run-name or ensure runs/ is not empty.")

    # 2. Construct path to the model file
    model_path = run_dir / "checkpoints" / args.model_file
    if not model_path.is_file():
        raise FileNotFoundError(f"[main] Model file {model_path} does not exist!")

    # 3. Collect input images
    if args.input is None:
        # By default, predict on images found in args.test_dir
        in_files = find_test_images(args.test_dir)
        if not in_files:
            raise FileNotFoundError(f"[main] No test images found in {args.test_dir}.")
    else:
        in_files = args.input

    # 4. Create a timestamped output folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path('output') / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # 4a. Create subfolders
    raw_output_dir = run_output_dir / 'raw'
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    viz_pred_dir = run_output_dir / 'viz_pred'
    viz_pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] Created subfolder for predicted overlays: {viz_pred_dir}")

    viz_pred_gt_dir = run_output_dir / 'viz_pred_gt'
    viz_pred_gt_dir.mkdir(parents=True, exist_ok=True)

    triple_plot_dir = run_output_dir / 'triple_plot'
    triple_plot_dir.mkdir(parents=True, exist_ok=True)

    # 4b. Write a small text file with info about the model & data
    info_file = run_output_dir / "run_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Model used: {model_path}\n")
        f.write(f"Test images path: {args.test_dir}\n")
        f.write(f"Ground truth path (if used): {args.gt_dir}\n")
        f.write(f"Date/Time: {timestamp}\n")
    print(f"[main] Wrote run information to: {info_file}")

    # 5. Build the UNet
    print("[main] Building UNet...")
    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[main] Using device: {device}")

    # Load the model weights
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)

    # Extract mask_values if stored
    mask_values = state_dict.pop('mask_values', [0, 1])

    net.load_state_dict(state_dict)
    print("[main] Model loaded successfully!")

    # 6. Determine output filenames
    out_files = get_output_filenames(args, in_files, raw_output_dir)

    # Step 7: Predict each test file
    for i, filename in tqdm(enumerate(in_files)):
        img = Image.open(filename)
        masks = predict_img(net, img, device=device, thresholds=args.mask_thresholds)

        for threshold, mask in masks.items():
            threshold_str = f"{int(threshold * 100)}"

            # Save mask
            if not args.no_save:
                mask_filename = raw_output_dir / f"{Path(filename).stem}_mask_{threshold_str}.png"
                mask_to_image(mask, mask_values).save(mask_filename)

            # Predicted overlay
            overlay_img_pred = overlay_prediction_on_image(
                img_pil=img, mask=mask, color=(255, 0, 255), alpha=0.3
            )
            overlay_pred_filename = viz_pred_dir / f"{Path(filename).stem}_overlay_pred_{threshold_str}.png"
            overlay_img_pred.save(overlay_pred_filename)

            # Overlay with ground truth (if available)
            gt_path = get_gt_filename(filename, args.gt_dir, suffix='_bolus')
            if gt_path and Path(gt_path).is_file():
                gt_img_pil = Image.open(gt_path)
                gt_arr = np.array(gt_img_pil, dtype=np.uint8)
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr[:, :, 0]

                overlay_gt_only = overlay_prediction_on_image(img_pil=img, mask=gt_arr, color=(0, 255, 0), alpha=0.3)

                threshold_overlays = {threshold: overlay_prediction_on_image(
                    img_pil=img, mask=mask, color=(255, 0, 255), alpha=0.3) for threshold, mask in masks.items()}

                threshold_comparison_plot_path = triple_plot_dir / f"{Path(filename).stem}_thresholds.png"
                create_threshold_comparison_plot(img, overlay_gt_only, threshold_overlays,
                                                 threshold_comparison_plot_path)

    print("[main] All predictions complete!")

def create_threshold_comparison_plot(original_img, overlay_gt_img, overlay_pred_dict, save_path):
    """
    Create a figure showing:
      - Left: Original image
      - Middle: Original image with ground truth overlay
      - Right columns: Predictions for multiple thresholds
    """
    num_thresholds = len(overlay_pred_dict)
    fig, axs = plt.subplots(1, num_thresholds + 2, figsize=(4 * (num_thresholds + 2), 4))

    axs[0].imshow(original_img)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(overlay_gt_img)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    for i, (threshold, overlay_pred_img) in enumerate(overlay_pred_dict.items()):
        axs[i + 2].imshow(overlay_pred_img)
        axs[i + 2].set_title(f"Prediction (th={threshold})")
        axs[i + 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    main()

'''

Foreback:
python .\predict.py --run-name run-20250202_084952 --model-file checkpoint_epoch3.pth --test-dir D:/Martin/data/foreback/processed/test/imgs   

'''