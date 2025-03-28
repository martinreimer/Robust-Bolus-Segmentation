import os
import shutil
from pathlib import Path

# Source and target directories
source_dir = Path(r"D:\Martin\thesis\data\raw\leonard")
target_dir = Path(r"D:\Martin\thesis\data\processed\leonard_dataset_final")

# Input folders
image_dir = source_dir / "image"
mask_dir = source_dir / "mask"

# Output structure
splits = ["train", "val", "test"]
for split in splits:
    (target_dir / split / "imgs").mkdir(parents=True, exist_ok=True)
    (target_dir / split / "masks").mkdir(parents=True, exist_ok=True)

# Get and sort all image IDs
all_ids = sorted([f.stem for f in image_dir.glob("*.png")], key=lambda x: int(x))
total = len(all_ids)

# Compute split indices
train_end = int(total * 0.7)
val_end = train_end + int(total * 0.2)

train_ids = all_ids[:train_end]
val_ids = all_ids[train_end:val_end]
test_ids = all_ids[val_end:]

def copy_files(ids, split):
    for id in ids:
        shutil.copy(image_dir / f"{id}.png", target_dir / split / "imgs" / f"{id}.png")
        shutil.copy(mask_dir / f"{id}.png", target_dir / split / "masks" / f"{id}.png")

# Copy files to new folders
copy_files(train_ids, "train")
copy_files(val_ids, "val")
copy_files(test_ids, "test")

print("Done. Dataset prepared at:", target_dir)
