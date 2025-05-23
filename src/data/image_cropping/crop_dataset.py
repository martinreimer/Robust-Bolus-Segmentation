#!/usr/bin/env python3
"""
Crop and copy an existing image+mask dataset into a new folder,
slicing each video’s frames to its union ROI across all frames, and forcing
a square crop by taking the larger side.

Usage:
  python crop_dataset.py \
      --original_dataset_root /path/to/original_dataset \
      --new_dataset_root    /path/to/cropped_dataset \
      [--bbox_scale_factor 1.0]

This script will:
  - Mirror the directory structure: train/imgs, train/masks, val/imgs, val/masks, test/imgs, test/masks
  - Compute a per-video union bounding box over all non-empty masks
  - Optionally expand that box by a scale factor
  - Enforce a square crop by enlarging the smaller dimension to match the larger one (centered)
  - Crop every frame and corresponding mask to that square ROI and save under new_dataset_root

Assumptions:
  - `data_overview.csv` lives at the original root and has columns: split, video_name, new_frame_name
  - Image files can be any extension; masks use the same basename plus `_bolus` suffix before the extension.
  - Splits are exactly `train`, `val`, `test`
"""
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image


def compute_mask_bounding_box(mask_path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    if ys.size == 0:
        return None
    return (xs.min(), ys.min(), xs.max(), ys.max())


def merge_bounding_boxes(bboxes):
    if not bboxes:
        return None
    xmins, ymins, xmaxs, ymaxs = zip(*bboxes)
    return (min(xmins), min(ymins), max(xmaxs), max(ymaxs))


def expand_bbox(bbox, img_w, img_h, scale_w=1.0, scale_h=1.0):
    if scale_w == 1.0 and scale_h == 1.0:
        return bbox
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    cx, cy = xmin + w/2, ymin + h/2
    nw, nh = w * scale_w, h * scale_h
    nxmin = int(max(0, cx - nw/2))
    nymin = int(max(0, cy - nh/2))
    nxmax = int(min(img_w, cx + nw/2))
    nymax = int(min(img_h, cy + nh/2))
    return (nxmin, nymin, nxmax, nymax)



def get_mask_path(orig_root, split, frame_name):
    mask_name = f"{frame_name}_bolus.png"
    return os.path.join(orig_root, split, "masks", mask_name), mask_name


def make_square(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    side = max(w, h)
    cx, cy = xmin + w/2, ymin + h/2
    nxmin = int(max(0, cx - side/2))
    nymin = int(max(0, cy - side/2))
    nxmax = int(min(img_w, cx + side/2))
    nymax = int(min(img_h, cy + side/2))
    # In rare cases clamping may shrink one side; re-center if needed
    final_w = nxmax - nxmin
    final_h = nymax - nymin
    if final_w != side or final_h != side:
        # Adjust to ensure square: expand back if possible
        diff_w = side - final_w
        diff_h = side - final_h
        nxmin = max(0, nxmin - diff_w//2)
        nymin = max(0, nymin - diff_h//2)
        nxmax = min(img_w, nxmin + side)
        nymax = min(img_h, nymin + side)
    return (nxmin, nymin, nxmax, nymax)


def main(orig_root, new_root, scale_w, scale_h):
    # prepare new directory structure
    for split in ["train", "val", "test"]:
        for folder in ["imgs", "masks"]:
            os.makedirs(os.path.join(new_root, split, folder), exist_ok=True)

    # load overview CSV
    csv_path = os.path.join(orig_root, "data_overview.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"data_overview.csv not found in {orig_root}")
    df = pd.read_csv(csv_path)

    # process per video
    for (split, video), group in df.groupby(["split", "video_name"]):
        if split not in ["train","val","test"]:
            continue

        # collect all mask bboxes
        bboxes = []
        for frame in group["frame_idx"]:
            mask_path, _ = get_mask_path(orig_root, split, frame)
            if os.path.isfile(mask_path):
                bb = compute_mask_bounding_box(mask_path)
                if bb:
                    bboxes.append(bb)

        # open first image for size
        first_frame = str(group["frame_idx"].iloc[0])
        img0 = Image.open(os.path.join(orig_root, split, "imgs", f"{first_frame}.png"))
        img_w, img_h = img0.size

        # union or full-frame
        video_bbox = merge_bounding_boxes(bboxes) or (0, 0, img_w, img_h)
        if not bboxes:
            print(f"[{split}] Video '{video}' has no masks; copying full frames.")
        else:
            print(f"[{split}] Video '{video}' ROI: {video_bbox}")

        # expand and square
        expanded = expand_bbox(video_bbox, img_w, img_h, scale_w=scale_w, scale_h=scale_h)
        crop_box = make_square(expanded, img_w, img_h)
        print(f"[{split}] Square crop: {crop_box}\n")

        # crop & copy all
        for frame in group["frame_idx"]:
            # image
            src_img = os.path.join(orig_root, split, "imgs", f"{frame}.png")
            dst_img = os.path.join(new_root, split, "imgs", f"{frame}.png")
            if os.path.isfile(src_img):
                Image.open(src_img).crop(crop_box).save(dst_img)
            # mask
            mask_path, mask_name = get_mask_path(orig_root, split, frame)
            dst_mask = os.path.join(new_root, split, "masks", mask_name)
            if os.path.isfile(mask_path):
                Image.open(mask_path).crop(crop_box).save(dst_mask)

        print(f"[{split}] Cropped and copied '{video}' ({len(group)} frames).\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_root", required=True,
                        help="Path to original dataset root")
    parser.add_argument("--new_dataset_root", required=True,
                        help="Path to write cropped dataset")
    parser.add_argument("--bbox_scale_width", type=float, default=1.0,
                        help="Width scale factor for ROI bbox expansion")
    parser.add_argument("--bbox_scale_height", type=float, default=1.0,
                        help="Height scale factor for ROI bbox expansion")
    args = parser.parse_args()
    main(args.original_dataset_root, args.new_dataset_root, args.bbox_scale_width, args.bbox_scale_height)

    '''
    
    python crop_dataset.py --original_dataset_root D:/Martin/thesis/data/processed/dataset_normal_0514_final --new_dataset_root D:/Martin/thesis/data/processed/dataset_normal_0514_final_roi_crop --bbox_scale_height 1.1 --bbox_scale_width 1.5

    python resize_images.py -p "D:\Martin\thesis\data\processed\dataset_normal_0514_final_roi_crop\train" --folders imgs masks -size 1024 -m pad_resize --in_place --only_stats
    '''