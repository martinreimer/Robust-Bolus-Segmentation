#region to be specified by user
DATA_EXPORT_PATH_1 = "../../data/raw/labelbox_output_mbs_0123"
DATA_EXPORT_PATH_2 = "../../data/raw/labelbox_output_mbss_martin_0123"
DATA_FOLDERS = [DATA_EXPORT_PATH_1, DATA_EXPORT_PATH_2]

UNIFIED_DATASET_PATH = "../../data/processed/dataset_first_experiments"

# Test size for train-test split
TEST_SIZE = 0.2

# Define target image resolution (e.g., 256x256)
TARGET_RESOLUTION = "512x512" # width x height
#endregion

"""
This script processes data from multiple source folders, creates a unified dataset, 
performs a train-test split based on video names, and organizes the data into 
appropriate directories. 
"""

import os
from tqdm import tqdm
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image


# Resize images with correct interpolation
def resize_image(img_path, target_width, target_height, save_path, is_mask=False):
    try:
        img = Image.open(img_path)
        if is_mask:
            # Use nearest-neighbor for masks
            img = img.resize((target_width, target_height), Image.NEAREST)
        else:
            # Use a smoother interpolation for normal images
            img = img.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
        img.save(save_path)
    except (FileNotFoundError, OSError):
        print(f"Error resizing image: {img_path}")



# create the dataset folder
os.makedirs(UNIFIED_DATASET_PATH, exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "imgs"), exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "masks"), exist_ok=True)

# Create train and test subfolders
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "train"), exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "train", "imgs"), exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "train", "masks"), exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "test"), exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "test", "imgs"), exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DATASET_PATH, "test", "masks"), exist_ok=True)

# go through the data folders and create the dataset
new_id = 0
data_overview = pd.DataFrame(columns=["frame_idx", "video_name", "frame", "new_frame_name", "new_mask_name", "source"])

for data_folder in DATA_FOLDERS:
    print(f"Processing data from {data_folder}...")
    # open data_overview.csv of data_folder
    old_data_overview_path = os.path.join(data_folder, "data_overview.csv")
    df_old_overview = pd.read_csv(old_data_overview_path)

    for index, row in tqdm(df_old_overview.iterrows()):
        old_frame_name = str(row["frame_idx"]) + ".png"
        old_mask_name = str(row["frame_idx"]) + "_bolus.png"

        # Check if mask file exists
        old_img_path = os.path.join(data_folder, "frames", old_frame_name)
        old_mask_path = os.path.join(data_folder, "masks", old_mask_name)


        # Copy image and mask to dataset folder
        new_frame_name = str(new_id) + ".png"
        new_mask_name = str(new_id) + "_bolus.png"
        img_path = os.path.join(UNIFIED_DATASET_PATH, "imgs", new_frame_name)
        mask_path = os.path.join(UNIFIED_DATASET_PATH, "masks", new_mask_name)

        shutil.copy(old_img_path, img_path)

        # If no mask (no bolus) exists, create a black mask
        if not os.path.exists(old_mask_path):
            new_mask = Image.new("L", (int(TARGET_RESOLUTION.split('x')[0]), int(TARGET_RESOLUTION.split('x')[1])), color=0)
            new_mask.save(mask_path)
        else:
            shutil.copy(old_mask_path, mask_path)

        # Get image resolution
        try:
            img = Image.open(img_path)
            width, height = img.size
            resolution = f"{width}x{height}"
        except (FileNotFoundError, OSError):
            resolution = "Resolution_Error"

        # Resize image
        resize_image(img_path, int(TARGET_RESOLUTION.split('x')[0]), int(TARGET_RESOLUTION.split('x')[1]),
                     img_path, is_mask=False)
        resize_image(mask_path, int(TARGET_RESOLUTION.split('x')[0]), int(TARGET_RESOLUTION.split('x')[1]),
                     mask_path, is_mask=True)

        # Add to data_overview using pd.concat()
        new_row = pd.DataFrame({
            "frame_idx": [row["frame_idx"]],
            "video_name": [row["video_name"]],
            "frame": [row["frame"]],
            "new_frame_name": [new_frame_name],
            "new_mask_name": [new_mask_name],
            "source": [data_folder],
            "resolution": [resolution]
        })
        data_overview = pd.concat([data_overview, new_row], ignore_index=True)

        new_id += 1

# Print resolution value counts
print("\nResolution Value Counts:")
print(data_overview['resolution'].value_counts())

# Save the initial data_overview.csv
data_overview.to_csv(os.path.join(UNIFIED_DATASET_PATH, "data_overview.csv"), index=False)

# Perform train-test split based on video names
train_videos, test_videos = train_test_split(data_overview["video_name"].unique(), test_size=TEST_SIZE, random_state=42)

# Create train and test DataFrames
train_df = data_overview[data_overview["video_name"].isin(train_videos)]
test_df = data_overview[data_overview["video_name"].isin(test_videos)]

# Save train and test data_overview.csv
train_df.to_csv(os.path.join(UNIFIED_DATASET_PATH, "train", "data_overview.csv"), index=False)
test_df.to_csv(os.path.join(UNIFIED_DATASET_PATH, "test", "data_overview.csv"), index=False)

# Copy images and masks to train and test folders
for _, row in train_df.iterrows():
    shutil.copy(os.path.join(UNIFIED_DATASET_PATH, "imgs", row["new_frame_name"]), os.path.join(UNIFIED_DATASET_PATH, "train", "imgs", row["new_frame_name"]))
    shutil.copy(os.path.join(UNIFIED_DATASET_PATH, "masks", row["new_mask_name"]), os.path.join(UNIFIED_DATASET_PATH, "train", "masks", row["new_mask_name"]))

for _, row in test_df.iterrows():
    shutil.copy(os.path.join(UNIFIED_DATASET_PATH, "imgs", row["new_frame_name"]), os.path.join(UNIFIED_DATASET_PATH, "test", "imgs", row["new_frame_name"]))
    shutil.copy(os.path.join(UNIFIED_DATASET_PATH, "masks", row["new_mask_name"]), os.path.join(UNIFIED_DATASET_PATH, "test", "masks", row["new_mask_name"]))


