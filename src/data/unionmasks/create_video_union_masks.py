# create_union_masks.py -------------------------------------------------------
import os, random, argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def load_mask(path):
    """Loads a PNG mask and returns a boolean array (True = foreground)."""
    m = Image.open(path).convert("L")
    return np.array(m) > 0

def union_masks(mask_paths):
    """OR-combines a list of masks -> uint8 image (0 or 255)."""
    if not mask_paths:
        return None
    union = None
    for p in mask_paths:
        if not os.path.exists(p):
            continue
        m = load_mask(p)
        union = m if union is None else union | m
    if union is None:
        return None
    return (union * 255).astype(np.uint8)  # convert bool->0/255

from PIL import Image

def overlay(pil_img, union_mask, color=(160, 32, 240), alpha=0.35):
    """
    Return pil_img with union_mask (uint8 0/255) overlaid in 'color'.
    Only mask pixels get tinted; background stays intact.
    """
    # Ensure 3-channel source
    base = pil_img.convert("RGBA")

    # 1) Solid-color layer
    overlay_rgba = Image.new("RGBA", base.size, color + (0,))

    # 2) Use the mask as alpha for that layer
    mask_alpha = Image.fromarray(union_mask, mode="L")
    overlay_rgba.putalpha(mask_alpha.point(lambda x: int(x * alpha)))

    # 3) Composite on top of the original
    combined = Image.alpha_composite(base, overlay_rgba)

    return combined.convert("RGB")


# --------------------------------------------------------------------------- #
def create_union_mask_dataset(
        original_root: str,
        new_root: str,
        num_frames_per_video: int = 10,
        union_dilate_px: int = 0  # optional extra dilation in pixels
    ):
    """
    Build a dataset where each sampled frame is paired with the *union mask*
    of all its video’s frames.
    """
    csv_path = os.path.join(original_root, "data_overview.csv")
    df       = pd.read_csv(csv_path)

    # make folders
    for split in ["train", "val", "test"]:
        for sub in ["images", "masks", "viz"]:
            os.makedirs(os.path.join(new_root, sub, split), exist_ok=True)

    # handle each (split, video_name) group
    for (split, vid), g in df.groupby(["split", "video_name"]):
        if split not in ["train", "val", "test"]:
            continue

        frame_rows = g.to_dict("records")

        # paths to every mask in this video
        mask_paths = []
        for r in frame_rows:
            base  = os.path.splitext(r["new_frame_name"])[0]
            mask_paths.append(os.path.join(original_root, split, "masks", f"{base}_bolus.png"))

        union = union_masks(mask_paths)
        if union is None:
            print(f"[{split}] {vid}: no foreground pixels -> skipped")
            continue

        # optional extra dilation (safety margin)
        if union_dilate_px > 0:
            from scipy.ndimage import binary_dilation
            union = binary_dilation(union > 0, iterations=union_dilate_px).astype(np.uint8) * 255

        # choose frames to copy
        chosen = frame_rows if len(frame_rows) <= num_frames_per_video \
                 else random.sample(frame_rows, k=num_frames_per_video)

        for row in chosen:
            fn  = row["new_frame_name"]
            img_src = os.path.join(original_root, split, "imgs", fn)
            if not os.path.exists(img_src):
                continue

            # save image
            img_dst = os.path.join(new_root, "images", split, fn)
            Image.open(img_src).save(img_dst)

            # save union mask (same resolution)
            mask_pil = Image.fromarray(union)
            mask_dst = os.path.join(new_root, "masks", split, fn)
            mask_pil.save(mask_dst)

            # save overlay visualisation
            viz = overlay(Image.open(img_src).convert("RGB"), union)
            viz_dst = os.path.join(new_root, "viz", split, fn)
            viz.save(viz_dst)

        print(f"[{split}] {vid}: saved {len(chosen)} frames with union mask.")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create union-mask dataset per video")
    parser.add_argument("--original_dataset_root", default="D:/path/to/original",
                        help="root folder containing train/val/test + data_overview.csv")
    parser.add_argument("--new_dataset_root", default="D:/path/to/new_union_mask_ds",
                        help="output root (images/, masks/, viz/)")
    parser.add_argument("--num_frames_per_video", type=int, default=20,
                        help="how many frames per video to copy")
    parser.add_argument("--mask_dilate_px", type=int, default=0,
                        help="extra dilation (pixels) to enlarge union mask")
    args = parser.parse_args()
    '''
python create_video_union_masks.py --original_dataset_root "D:/Martin/thesis/data/processed/dataset_0328_final" --new_dataset_root "D:/Martin/thesis/data/processed/dataset_0328_final_video_union" --num_frames_per_video  200
'''

    random.seed(42)
    create_union_mask_dataset(
        original_root=args.original_dataset_root,
        new_root=args.new_dataset_root,
        num_frames_per_video=args.num_frames_per_video,
        union_dilate_px=args.mask_dilate_px
    )
    print("Done.")
# --------------------------------------------------------------------------- #
