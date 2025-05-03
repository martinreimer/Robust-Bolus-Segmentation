import os
import sys
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

def compute_mask_bounding_box(mask_path):
    """
    Given a path to a mask (PNG), return (xmin, ymin, xmax, ymax)
    in pixel coordinates. If mask is fully empty, returns None.
    """
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask_arr = np.array(mask)

    # Find all non-zero (foreground) pixels
    foreground = np.where(mask_arr > 0)
    if foreground[0].size == 0:
        # No non-zero pixels
        return None

    ymin, ymax = foreground[0].min(), foreground[0].max()
    xmin, xmax = foreground[1].min(), foreground[1].max()

    return (xmin, ymin, xmax, ymax)

def merge_bounding_boxes(boxes):
    """
    Given a list of bounding boxes [(xmin, ymin, xmax, ymax), ...],
    return the union bounding box as (xmin, ymin, xmax, ymax).
    If boxes is empty, return None.
    """
    if not boxes:
        return None

    xmins = [b[0] for b in boxes]
    ymins = [b[1] for b in boxes]
    xmaxs = [b[2] for b in boxes]
    ymaxs = [b[3] for b in boxes]
    return (min(xmins), min(ymins), max(xmaxs), max(ymaxs))

def expand_bbox(bbox, image_width, image_height, scale_factor=1.0):
    """
    Expand (xmin, ymin, xmax, ymax) around its center by scale_factor.
    Clamps to image boundaries.
    """
    (xmin, ymin, xmax, ymax) = bbox
    if scale_factor == 1.0:
        return bbox  # No change

    # Current width/height
    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    # New width/height
    new_bw = bw * scale_factor
    new_bh = bh * scale_factor

    # Recompute corners, then clamp
    new_xmin = int(cx - new_bw / 2.0)
    new_xmax = int(cx + new_bw / 2.0)
    new_ymin = int(cy - new_bh / 2.0)
    new_ymax = int(cy + new_bh / 2.0)

    # Clamp
    new_xmin = max(0, new_xmin)
    new_ymin = max(0, new_ymin)
    new_xmax = min(image_width - 1, new_xmax)
    new_ymax = min(image_height - 1, new_ymax)

    return (new_xmin, new_ymin, new_xmax, new_ymax)

def yolo_v8_format(bbox, image_width, image_height):
    """
    Convert (xmin, ymin, xmax, ymax) in pixel coords to
    YOLOv8 format: (class, x_center, y_center, w, h),
    normalized [0..1].
    We'll hardcode 'class' = 0 for "ROI".
    """
    (xmin, ymin, xmax, ymax) = bbox
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin

    x_center = xmin + bbox_w / 2
    y_center = ymin + bbox_h / 2

    # Normalize
    x_center_n = x_center / image_width
    y_center_n = y_center / image_height
    w_n = bbox_w / image_width
    h_n = bbox_h / image_height

    return (0, x_center_n, y_center_n, w_n, h_n)

def draw_bbox_on_image(pil_image, bbox, color=(160, 32, 240), width=3):
    """
    Draw the bounding box on the image with the specified color and line width.
    Default color is a purple-like value (RGB).
    """
    draw = ImageDraw.Draw(pil_image)
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)
    return pil_image

def create_bbox_dataset(
    original_dataset_root,
    new_bbox_root,
    num_frames_per_video=10,
    bbox_scale_factor=1.0
):
    """
    Creates a bounding box dataset from an existing segmentation dataset by:
    - computing a single 'union' bounding box per video (all frames)
    - randomly sampling a specified number of frames per video
    - saving those frames (images, YOLO label txt, and a 'viz' image)

    Folder structure after creation (Ultralytics YOLOv8 style):
      new_bbox_root/
      ├── images
      │   ├── train
      │   ├── val
      │   └── test
      ├── labels
      │   ├── train
      │   ├── val
      │   └── test
      └── viz
          ├── train
          ├── val
          └── test

    :param original_dataset_root: Path to the original dataset root directory,
                                  which must contain:
                                  - data_overview.csv
                                  - train/val/test folders each with imgs/ and masks/
    :param new_bbox_root:        Path where the new bounding box dataset will be stored.
    :param num_frames_per_video: Number of random frames to sample per video (default=10).
    :param bbox_scale_factor:    Factor to expand bounding box around its center (default=1.0).
    """
    import shutil

    csv_path = os.path.join(original_dataset_root, "data_overview.csv")

    # Create subdirs for YOLO structure: images/{train,val,test}, labels/{train,val,test}, viz/{train,val,test}
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(new_bbox_root, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(new_bbox_root, "labels", subset), exist_ok=True)
        os.makedirs(os.path.join(new_bbox_root, "viz", subset), exist_ok=True)

    df = pd.read_csv(csv_path)

    # Group by (split, video_name)
    grouped = df.groupby(["split", "video_name"])

    for (split, video_name), group_df in grouped:
        # Only process if split is train/val/test
        if split not in ["train", "val", "test"]:
            continue

        # 1) Compute the union bounding box for this video
        all_bboxes = []
        frame_rows = group_df.to_dict("records")

        for row in frame_rows:
            frame_name = row["new_frame_name"]
            base_name = os.path.splitext(frame_name)[0]
            mask_name = base_name + "_bolus.png"

            mask_path = os.path.join(original_dataset_root, split, "masks", mask_name)
            if os.path.exists(mask_path):
                bbox = compute_mask_bounding_box(mask_path)
                if bbox is not None:
                    all_bboxes.append(bbox)

        video_bbox = merge_bounding_boxes(all_bboxes)
        if video_bbox is None:
            # means empty masks for the entire video, skip
            continue

        # 2) Sample up to num_frames_per_video from the frames in this video
        if len(frame_rows) <= num_frames_per_video:
            chosen_rows = frame_rows
        else:
            chosen_rows = random.sample(frame_rows, k=num_frames_per_video)

        # 3) For each chosen frame, save image, YOLO label, and visualization
        for row in chosen_rows:
            frame_name = row["new_frame_name"]
            img_path  = os.path.join(original_dataset_root, split, "imgs", frame_name)
            if not os.path.exists(img_path):
                print(f"Warning: image {img_path} not found, skipping.")
                continue

            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            # expand the union bbox
            expanded_bbox = expand_bbox(video_bbox, w, h, bbox_scale_factor)

            # YOLO format
            (class_id, x_center, y_center, w_norm, h_norm) = yolo_v8_format(expanded_bbox, w, h)

            # new paths (YOLO style)
            # images/<split>/<filename.png>
            # labels/<split>/<filename.txt>
            # viz/<split>/<filename.png>
            new_img_path   = os.path.join(new_bbox_root, "images", split, frame_name)
            label_filename = os.path.splitext(frame_name)[0] + ".txt"
            new_label_path = os.path.join(new_bbox_root, "labels", split, label_filename)
            new_viz_path   = os.path.join(new_bbox_root, "viz", split, frame_name)

            # save image
            image.save(new_img_path)

            # save label
            with open(new_label_path, "w") as f:
                # YOLOv8: class x_center y_center w h (normalized)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            # visualization
            viz_image = image.copy()
            viz_image = draw_bbox_on_image(viz_image, expanded_bbox, color=(160, 32, 240), width=3)
            viz_image.save(new_viz_path)

        print(f"[{split}] Video: {video_name} -> saved {len(chosen_rows)} frames with a union bbox.")

def main():
    """
    CLI usage:
      python create_bbox_dataset.py <original_dataset_root> <new_bbox_root> [num_frames_per_video] [bbox_scale_factor]

    Defaults:
      num_frames_per_video = 10
      bbox_scale_factor    = 1.0
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Create bounding box dataset from a segmentation dataset with random frame sampling."
    )
    parser.add_argument("--original_dataset_root", type=str, default="D:/Martin/thesis/data/processed/dataset_0328_final",
                        required=False, help="Path to the original dataset (must have train/val/test and data_overview.csv).")
    parser.add_argument("--new_bbox_root", type=str, default="D:/Martin/thesis/data/processed/dataset_0328_roi_detection_final",
                        required=False, help="Where to store the new bounding box dataset.")
    parser.add_argument("--num_frames_per_video", type=int, default=20, required=False,
                        help="How many random frames to sample per video (default=10).")
    parser.add_argument("--bbox_scale_factor", type=float, default=1.2, required=False,
                        help="Scale factor to expand bounding boxes around their center (default=1.0).")

    args = parser.parse_args()

    # Create the dataset
    random.seed(42)  # for reproducibility if desired
    create_bbox_dataset(
        original_dataset_root=args.original_dataset_root,
        new_bbox_root=args.new_bbox_root,
        num_frames_per_video=args.num_frames_per_video,
        bbox_scale_factor=args.bbox_scale_factor
    )
    print("Done.")


if __name__ == "__main__":
    main()
    '''
    python create_yolo_dataset.py --num_frames_per_video 20 --bbox_scale_factor 1.2
    
    python create_yolo_dataset.py /path/to/segmentation_dataset /path/to/new_bbox_dataset \
        --num_frames_per_video 20 --bbox_scale_factor 1.2

    '''