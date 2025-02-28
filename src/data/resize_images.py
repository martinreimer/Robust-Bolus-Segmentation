import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
from torchvision import transforms

def collect_image_paths(folder_path):
    """
    Collects image paths from the specified folder.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a valid directory.")
    return sorted(folder.glob("*.png"))

def resize_and_center_crop(image_path, target_size, is_mask):
    """
    Resize image so that the shorter side is target_size while keeping aspect ratio,
    then center crop to target_size x target_size.
    """
    image = Image.open(image_path).convert("L" if is_mask else "RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size)
    ])
    return transform(image)

def resize_and_pad(image_path, target_size, is_mask, pad_value=0):
    """
    Resize image while preserving aspect ratio, then pad to target_size x target_size.
    """
    image = Image.open(image_path).convert("L" if is_mask else "RGB")
    w, h = image.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.NEAREST if is_mask else Image.Resampling.LANCZOS)
    pad_image = Image.new("L" if is_mask else "RGB", (target_size, target_size), pad_value)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    pad_image.paste(image, (paste_x, paste_y))
    return pad_image

def pad_then_resize(image_path, target_size, is_mask, pad_value=0):
    """
    Pad image to a square first, then resize to target_size x target_size.
    """
    image = Image.open(image_path).convert("L" if is_mask else "RGB")
    w, h = image.size
    max_dim = max(w, h)
    pad_image = Image.new("L" if is_mask else "RGB", (max_dim, max_dim), pad_value)
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    pad_image.paste(image, (paste_x, paste_y))
    return pad_image.resize((target_size, target_size), Image.NEAREST if is_mask else Image.Resampling.LANCZOS)

def process_folder(folder_path, target_size, is_mask, only_stats, in_place, method, pad_value=0):
    """
    Processes images in the given folder using the specified resizing method.
    """
    image_paths = collect_image_paths(folder_path)
    resolutions = [Image.open(p).size for p in image_paths]
    resolution_counts = Counter(resolutions)

    print(f"Before processing stats for {folder_path}:")
    for res, count in resolution_counts.items():
        print(f"Resolution {res}: {count} files")

    if not only_stats:
        output_folder = folder_path if in_place else Path(f"{folder_path}_processed")
        output_folder.mkdir(exist_ok=True, parents=True)

        for image_path in image_paths:
            if method == "center_crop":
                processed_image = resize_and_center_crop(image_path, target_size, is_mask)
            elif method == "resize_pad":
                processed_image = resize_and_pad(image_path, target_size, is_mask, pad_value)
            elif method == "pad_resize":
                processed_image = pad_then_resize(image_path, target_size, is_mask, pad_value)

            save_path = image_path if in_place else output_folder / image_path.name
            processed_image.save(save_path)
        print(f"Processing complete for {folder_path}")

def main():
    parser = argparse.ArgumentParser(description="Resize images and masks using different strategies.")
    parser.add_argument('--dataset_path', '-p', required=True, help="Path to the dataset directory")
    parser.add_argument('--target_size', '-size', type=int, required=False, default=256, help="Target size for resizing")
    parser.add_argument('--folders', '-f', nargs='+', default=['imgs', 'masks'], help="Folders to process (default: imgs masks)")
    parser.add_argument('--method', '-m', choices=['center_crop', 'resize_pad', 'pad_resize'], required=True, help="Resizing method to use")
    parser.add_argument('--pad_value', '-pv', type=int, default=0, help="Padding value (0 for black, 255 for white)")
    parser.add_argument('--only_stats', action='store_true', help="Only collect resolution statistics without resizing")
    parser.add_argument('--in_place', action='store_true', help="Modify images in place instead of saving in a new folder")
    args = parser.parse_args()

    for folder in args.folders:
        folder_path = Path(args.dataset_path) / folder
        if folder_path.exists():
            is_mask = 'masks' in folder.lower()
            print(f"Processing folder {folder_path} (is mask: {is_mask})")
            process_folder(folder_path, args.target_size, is_mask, args.only_stats, args.in_place, args.method, args.pad_value)
        else:
            print(f"Warning: Folder {folder_path} does not exist.")
    print("Processing complete.")

if __name__ == "__main__":
    main()
# python resize_images.py -p ../../data/foreback/processed/val --folders imgs masks -size 512 -m pad_resize --in_place