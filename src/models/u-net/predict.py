import argparse
import logging
import os
import re
from pathlib import Path
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def find_runs_dir(run_name=None, runs_root='runs'):
    """
    Finds a run directory under 'runs/' either by:
      - run_name (exact subdir match), or
      - the most recently modified run if run_name is None.
    Returns the Path to that run directory, or None if none found.
    """
    runs_root = Path(runs_root)

    if run_name:
        # User explicitly provided a run name
        candidate = runs_root / run_name
        if candidate.is_dir():
            return candidate
        else:
            logging.warning(f"Could not find run directory {candidate}, returning None.")
            return None
    else:
        # Pick the newest directory that starts with 'run-'
        run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith('run-')]
        if not run_dirs:
            logging.warning(f"No run directories found in {runs_root}")
            return None

        # Sort by modification time descending
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return run_dirs[0]  # newest


def predict_img(net,
                full_img: Image.Image,
                device: torch.device,
                scale_factor=1.0,
                out_threshold=0.5):
    """Predict mask for a single image `full_img` using a trained `net`."""
    net.eval()

    if net.n_channels == 1:
        # convert to grayscale
        full_img = full_img.convert('L')
    else:
        # assume 3 channels => convert to RGB
        full_img = full_img.convert('RGB')

    # Convert image to np.float32
    img_np = np.array(full_img, dtype=np.float32)
    print("hey")
    print(img_np.shape)  # HxWxC
    # Preprocess using the BasicDataset method => shape (C, scaledH, scaledW)
    img_np = BasicDataset.preprocess(img_np, scale_factor, is_mask=False)
    if net.n_channels == 1 and img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=0)

    elif net.n_channels == 3 and img_np.ndim == 3:
        # If your preprocess yields shape (H, W, 3), you might need to transpose to (3, H, W).
        img_np = np.transpose(img_np, (2, 0, 1))

    # Now img_np is (C, H, W). Next we unsqueeze batch => (1, C, H, W)
    img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device, dtype=torch.float32)
    print(img_torch.shape)  # should be [1, 1, H, W] or [1, 3, H, W]


    # Move to torch tensor => shape (1, C, scaledH, scaledW)
    img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    #img dims
    print(img_torch.shape)
    with torch.no_grad():
        output = net(img_torch).cpu()
        # Resize the output back to the original image shape (H, W)
        orig_size = (full_img.size[1], full_img.size[0])  # (height, width)
        output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=False)

        if net.n_classes > 1:
            # multi-class => argmax
            mask = output.argmax(dim=1)  # [1, H, W]
        else:
            # binary => sigmoid + threshold
            mask = (torch.sigmoid(output) > out_threshold)  # [1, 1, H, W]

    # Return [H, W] or [1, H, W] -> but typically we want a 2D np.array
    return mask[0].long().squeeze().numpy()


def mask_to_image(mask: np.ndarray, mask_values):
    """
    Convert the numeric mask to a PIL image based on `mask_values`.
    For example, if mask_values=[0,255], we map 0->0, 1->255 for binary.
    If multi-class with e.g. mask_values=[0,127,255], we map class 0->0, class 1->127, class 2->255, etc.
    """
    # mask shape is (H, W)
    if len(mask_values) == 2 and set(mask_values) == {0, 1}:
        # interpret as boolean
        out = np.zeros(mask.shape, dtype=np.uint8)
        out[mask == 1] = 255
        return Image.fromarray(out)

    # Otherwise, create an 8-bit array
    out = np.zeros(mask.shape, dtype=np.uint8)
    for i, val in enumerate(mask_values):
        out[mask == i] = val
    return Image.fromarray(out)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name of a specific run directory in "runs/". If not specified, use the newest run.')
    parser.add_argument('--runs-root', type=str, default='runs',
                        help='Parent directory containing run subdirs, default="runs"')
    parser.add_argument('--model-file', type=str, default='model.pth',
                        help='Name of the model file in the run directory, default="model.pth"')

    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='Filenames of input images. Default is all images in ./data/test',
                        default=None)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Filenames of output images. If not given, automatically generated')

    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')

    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white (for binary)')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images. (If you trained with 0.5, you can do the same here.)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes (1 for binary, else >1)')
    parser.add_argument('--channels', type=int, default=1, help='Number of input channels (default=1 for grayscale)')

    return parser.parse_args()


def get_output_filenames(args, in_files):
    """
    If user gave --output, we use it directly.
    Otherwise, for each input file we create something like "output/raw/<stem>_OUT.png".
    """
    if args.output is not None:
        # If user specified output paths => use them
        return args.output
    else:
        # By default, store in "output/raw/<stem>_OUT.png"
        out_dir = Path('output') / 'raw'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_files = []
        for fn in in_files:
            stem = Path(fn).stem
            out_files.append(str(out_dir / f"{stem}_OUT.png"))
        return out_files


def find_test_images(test_dir='data/test'):
    """
    If user doesn't specify input images, we default to all images in ./data/test
    that have typical image suffixes.
    """
    test_dir_path = Path(test_dir)
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = []
    for ext in valid_exts:
        files.extend(test_dir_path.glob(f'*{ext}'))
    # Convert to string sorted list
    files = sorted(str(f) for f in files)
    return files


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Determine the run directory
    run_dir = find_runs_dir(run_name=args.run_name, runs_root=args.runs_root)
    if run_dir is None:
        raise FileNotFoundError("No valid run directory found. Either specify --run-name or ensure runs/ is not empty.")

    # 2. Construct path to the model file
    model_path = run_dir / "checkpoints" / args.model_file
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file {model_path} does not exist!")

    # 3. Collect input images
    if args.input is None:
        # By default, predict on data/test/ images
        in_files = find_test_images('data/test/imgs')
        if not in_files:
            raise FileNotFoundError("No test images found in ./data/test.")
    else:
        in_files = args.input

    # 4. Create output filenames
    out_files = get_output_filenames(args, in_files)

    # 5. Build the UNet with desired channels/classes
    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model from {model_path}')
    logging.info(f'Using device: {device}')

    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)

    # Extract mask_values if stored
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    # 6. Predict each file
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image: {filename}')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # 7. Save result if desired
        if not args.no_save:
            out_filename = out_files[i]
            # Convert mask to actual image
            result_img = mask_to_image(mask, mask_values)
            result_img.save(out_filename)
            logging.info(f"Mask saved to: {out_filename}")

        if args.viz:
            from utils.utils import plot_img_and_mask
            logging.info(f'Visualizing results for image {filename}, close the figure to continue...')
            plot_img_and_mask(img, mask)

if __name__ == '__main__':
    main()
