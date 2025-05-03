import os
import shutil
from PIL import Image


def zoom_center(image, zoom_ratio):
    """
    Zoom into the center of the image by a given zoom_ratio.
    zoom_ratio < 1 => zoom in
    zoom_ratio = portion of original dimension we keep from the center,
    and then we resize back to the original size.

    :param image: PIL.Image object
    :param zoom_ratio: float, 0 < zoom_ratio <= 1
    :return: zoomed PIL.Image object, same size as original
    """
    if not (0 < zoom_ratio <= 1):
        raise ValueError("zoom_ratio must be in the range (0, 1].")

    # Original size
    width, height = image.size

    # Calculate cropping area
    crop_width = int(width * zoom_ratio)
    crop_height = int(height * zoom_ratio)

    # Coordinates for center crop
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # Crop the image
    cropped_img = image.crop((left, top, right, bottom))

    # Resize back to original
    zoomed_img = cropped_img.resize((width, height), Image.LANCZOS)
    return zoomed_img


def copy_and_zoom_dataset(
        original_root,
        new_root,
        zoom_ratio=0.8
):
    """
    Copy dataset directory structure from `original_root` to `new_root`,
    and zoom in images and masks by the specified zoom_ratio.

    :param original_root: Path to the original dataset (containing train/val/test).
    :param new_root: Path where the new zoomed dataset will be saved.
    :param zoom_ratio: float in (0, 1], how much of the center we keep.
    """
    # Create the new root directory if it doesn't exist
    os.makedirs(new_root, exist_ok=True)

    # The expected subdirs are train, val, test
    subsets = ["train", "val", "test"]
    for subset in subsets:
        subset_path = os.path.join(original_root, subset)
        if not os.path.isdir(subset_path):
            # If a subset doesn't exist in your dataset, skip it
            continue

        # We expect 'imgs' and 'masks' in each subset
        for subfolder in ["imgs", "masks"]:
            original_subfolder_path = os.path.join(subset_path, subfolder)
            if not os.path.isdir(original_subfolder_path):
                continue

            # Make the corresponding directory in the new dataset
            new_subfolder_path = os.path.join(new_root, subset, subfolder)
            os.makedirs(new_subfolder_path, exist_ok=True)

            # Loop through each file in imgs/masks
            for filename in os.listdir(original_subfolder_path):
                # Only process common image formats
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    continue

                old_filepath = os.path.join(original_subfolder_path, filename)
                new_filepath = os.path.join(new_subfolder_path, filename)

                # Open the image
                image = Image.open(old_filepath)

                # Zoom the image
                zoomed = zoom_center(image, zoom_ratio)

                # Save to new path
                zoomed.save(new_filepath)

    print(f"Finished creating zoomed dataset at: {new_root}")


if __name__ == "__main__":
    # Example usage:
    original_dataset_path = "D:/Martin/thesis/data/processed/dataset_0328_final"#D:\Martin\thesis\data\processed\dataset_0328_final\train\imgs
    new_dataset_path = "D:/Martin/thesis/data/processed/dataset_0328_final_zoom_80"

    # create new dataset folder if it doesn't exist
    os.makedirs(new_dataset_path)
    # Adjust this zoom_ratio as needed
    zoom_ratio = 0.8

    copy_and_zoom_dataset(
        original_root=original_dataset_path,
        new_root=new_dataset_path,
        zoom_ratio=zoom_ratio
    )
