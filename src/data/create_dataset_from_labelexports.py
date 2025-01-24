"""
create_unified_dataset.py

This script reads data from multiple source folders (each containing a data_overview.csv),
copies/resizes images and masks into a single 'unified' dataset folder with subfolders
"imgs" and "masks". It outputs a single 'data_overview.csv' that references all images.
It does NOT perform any train/test splitting.
"""

import os
import shutil
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

# 3rd-party for resizing
# from PIL import Image
# e.g. pip install pillow

# -----------------------
# --- USER PARAMETERS ---
# -----------------------
DATA_EXPORT_PATH_1 = "../../data/raw/labelbox_output_mbs_0123"
DATA_EXPORT_PATH_2 = "../../data/raw/labelbox_output_mbss_martin_0123"
DATA_FOLDERS = [DATA_EXPORT_PATH_1, DATA_EXPORT_PATH_2]

UNIFIED_DATASET_PATH = "../../data/processed/dataset_labelbox_export_0124"

# Define target image resolution (width x height)
TARGET_RESOLUTION = "256x256"  # e.g. "512x512"

# --------------------
# --- MAIN LOGIC  ---
# --------------------

def resize_image(img_path, target_width, target_height, save_path, is_mask=False):
    """
    Resizes an image to target_width x target_height and saves it to save_path.
    Uses nearest-neighbor for masks, LANCZOS (or similar) for regular images.
    """
    try:
        img = Image.open(img_path)
        if is_mask:
            img = img.resize((target_width, target_height), Image.NEAREST)
        else:
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img.save(save_path)
    except (FileNotFoundError, OSError):
        print(f"[resize_image] Error resizing image: {img_path}")


def main():
    # 1. Create the unified dataset structure (imgs/ and masks/)
    os.makedirs(UNIFIED_DATASET_PATH, exist_ok=True)
    imgs_dir = os.path.join(UNIFIED_DATASET_PATH, "imgs")
    masks_dir = os.path.join(UNIFIED_DATASET_PATH, "masks")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # 2. Prepare a DataFrame to collect info about all images/masks
    data_overview = pd.DataFrame(columns=[
        "frame_idx",
        "video_name",
        "frame",
        "new_frame_name",
        "new_mask_name",
        "source",
        "resolution"
    ])

    # 3. Start copying from source folders
    new_id = 0
    target_w, target_h = map(int, TARGET_RESOLUTION.split('x'))

    for data_folder in DATA_FOLDERS:
        print(f"\n[main] Processing data from {data_folder} ...")
        old_data_overview_path = os.path.join(data_folder, "data_overview.csv")
        if not os.path.isfile(old_data_overview_path):
            print(f"Warning: Could not find data_overview.csv in {data_folder}. Skipping.")
            continue

        df_old = pd.read_csv(old_data_overview_path)

        for _, row in tqdm(df_old.iterrows(), total=len(df_old)):
            old_frame_idx = row["frame_idx"]
            old_frame_name = f"{old_frame_idx}.png"
            old_mask_name = f"{old_frame_idx}_bolus.png"

            old_img_path = os.path.join(data_folder, "frames", old_frame_name)
            old_mask_path = os.path.join(data_folder, "masks", old_mask_name)

            # Construct new filenames
            new_frame_name = f"{new_id}.png"
            new_mask_name = f"{new_id}_bolus.png"
            new_img_path = os.path.join(imgs_dir, new_frame_name)
            new_msk_path = os.path.join(masks_dir, new_mask_name)

            # Copy or create a black mask if mask doesn't exist
            shutil.copy(old_img_path, new_img_path)
            if os.path.exists(old_mask_path):
                shutil.copy(old_mask_path, new_msk_path)
            else:
                # create a black mask
                blank_mask = Image.new("L", (target_w, target_h), color=0)
                blank_mask.save(new_msk_path)

            # Attempt to get original resolution
            try:
                img = Image.open(new_img_path)
                w, h = img.size
                original_res = f"{w}x{h}"
            except:
                original_res = "Resolution_Error"

            # Resize the images in-place
            resize_image(new_img_path, target_w, target_h, new_img_path, is_mask=False)
            resize_image(new_msk_path, target_w, target_h, new_msk_path, is_mask=True)

            # Add new row to our master overview
            new_row = {
                "frame_idx": row["frame_idx"],
                "video_name": row["video_name"],
                "frame": row["frame"],
                "new_frame_name": new_frame_name,
                "new_mask_name": new_mask_name,
                "source": data_folder,
                "resolution": original_res,
            }
            data_overview = pd.concat([data_overview, pd.DataFrame([new_row])], ignore_index=True)

            new_id += 1

    # Print resolution stats
    print("\n[main] Resolution Value Counts in final dataset:")
    print(data_overview["resolution"].value_counts())

    # 4. Save data_overview.csv (no train/test split here!)
    overview_csv_path = os.path.join(UNIFIED_DATASET_PATH, "data_overview.csv")
    data_overview.to_csv(overview_csv_path, index=False)
    print(f"[main] Wrote unified dataset overview to {overview_csv_path}")

    print("\n[main] Done creating unified dataset!")


if __name__ == "__main__":
    main()
