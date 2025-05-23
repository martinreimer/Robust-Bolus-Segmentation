#!/usr/bin/env python
"""
Script to generate per-video visualization videos by overlaying masks on frames.

Usage:
    python visualize_videos.py --processed_dir /path/to/processed/output
                              [--fps 10]
                              [--mask_color 255,0,255]  # RGB purple
                              [--alpha 0.2]
"""
import os
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image


def overlay_mask_on_image(img_pil: Image.Image,
                          mask: np.ndarray,
                          mask_color=(255, 0, 255),  # Purple
                          alpha=0.2) -> Image.Image:
    """
    Overlay a ground truth mask on the original image.
    """
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    base_np = np.array(img_pil)
    mask_overlay = np.zeros_like(base_np)
    if mask.shape != base_np.shape[:2]:
        print(f"Error: Mask shape {mask.shape} does not match image shape {base_np.shape}")
        return img_pil
    mask_overlay[mask != 0] = mask_color

    # Blend
    blended = (alpha * mask_overlay + (1 - alpha) * base_np).astype(np.uint8)
    return Image.fromarray(blended)


def create_visualizations(processed_dir: str,
                          fps: int,
                          mask_color: tuple[int, int, int],
                          alpha: float):
    # Paths
    overview_csv = os.path.join(processed_dir, 'data_overview.csv')
    imgs_dir = os.path.join(processed_dir, 'imgs')
    masks_dir = os.path.join(processed_dir, 'masks')
    viz_dir = os.path.join(processed_dir, 'video_viz')
    os.makedirs(viz_dir, exist_ok=True)

    # Load overview
    df = pd.read_csv(overview_csv)
    # Ensure sorted by frame index
    df['frame_idx'] = df['frame_idx'].astype(int)
    grouped = df.groupby('shared_video_id')

    for vid, group in grouped:
        group_sorted = group.sort_values('frame_idx')
        # Determine video size from first frame
        first_idx = group_sorted.iloc[0]['frame_idx']
        # try jpg first, then png
        #check if exists
        if os.path.exists(os.path.join(imgs_dir, f"{first_idx}.jpg")):
            first_img_path = os.path.join(imgs_dir, f"{first_idx}.jpg")
        elif os.path.exists(os.path.join(imgs_dir, f"{first_idx}.png")):
            first_img_path = os.path.join(imgs_dir, f"{first_idx}.png")
        else:
            print(f"Warning: No image found for video {vid}, skipping.")
            continue

        img0 = Image.open(first_img_path)
        width, height = img0.size

        # VideoWriter expects (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(viz_dir, f"{vid}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for _, row in group_sorted.iterrows():
            idx = int(row['frame_idx'])
            # check if jpg first, then png
            if os.path.exists(os.path.join(imgs_dir, f"{idx}.jpg")):
                frame_path = os.path.join(imgs_dir, f"{idx}.jpg")
                mask_path = os.path.join(masks_dir, f"{idx}_bolus.jpg")
            elif os.path.exists(os.path.join(imgs_dir, f"{idx}.png")):
                frame_path = os.path.join(imgs_dir, f"{idx}.png")
                mask_path = os.path.join(masks_dir, f"{idx}_bolus.png")
            else:
                print(f"Warning: No image found for frame {idx}, skipping.")
                continue
            img_pil = Image.open(frame_path)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros((img_pil.height, img_pil.width), dtype=np.uint8)

            overlay = overlay_mask_on_image(img_pil, mask, mask_color, alpha)
            # Convert to BGR for OpenCV
            overlay_np = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
            writer.write(overlay_np)

        writer.release()
        print(f"Generated visualization video: {out_path}")


def parse_color(color_str: str):
    parts = [int(p) for p in color_str.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("mask_color must be R,G,B")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-video overlay visualizations from processed dataset.")
    parser.add_argument("--processed_dir", required=True,
                        help="Path to the processed dataset root (contains imgs, masks, data_overview.csv)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for output videos")
    parser.add_argument("--mask_color", type=parse_color, default="255,0,255",
                        help="RGB color for mask overlay, as R,G,B")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Transparency for mask overlay (0.0-1.0)")
    args = parser.parse_args()

    create_visualizations(
        processed_dir=args.processed_dir,
        fps=args.fps,
        mask_color=args.mask_color,
        alpha=args.alpha
    )

if __name__ == '__main__':
    main()
