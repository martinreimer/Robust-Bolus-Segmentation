import os
import numpy as np
from PIL import Image
from torchvision import transforms


def resize_and_center_crop_images(input_folder, output_folder, sample_size=10):
    """
    Resize images in input_folder such that the shorter side is 256 while keeping aspect ratio,
    then center crop to 256x256, and save them in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:sample_size]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256)
    ])

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            processed_image = transform(image)
            processed_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")


def resize_and_pad_images(input_folder, output_folder, sample_size=10, pad_value=0):
    """
    Resize images in input_folder such that the shorter side is 256 while keeping aspect ratio,
    then pad to 256x256, and save them in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:sample_size]

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        try:
            image = Image.open(img_path).convert("L")
            w, h = image.size
            scale = 256 / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.NEAREST)

            pad_image = Image.new("L", (256, 256), pad_value)
            paste_x = (256 - new_w) // 2
            paste_y = (256 - new_h) // 2
            pad_image.paste(image, (paste_x, paste_y))
            pad_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")


def pad_then_resize_images(input_folder, output_folder, sample_size=10, pad_value=0):
    """
    Pad images in input_folder to the largest dimension,
    then resize to 256x256 while maintaining binary mask integrity.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:sample_size]

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        try:
            image = Image.open(img_path).convert("L")
            w, h = image.size
            max_dim = max(w, h)
            pad_image = Image.new("L", (max_dim, max_dim), pad_value)
            paste_x = (max_dim - w) // 2
            paste_y = (max_dim - h) // 2
            pad_image.paste(image, (paste_x, paste_y))

            resized_image = pad_image.resize((256, 256), Image.NEAREST)
            resized_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")


# Paths
input_folder_imgs = "../../data/foreback/processed/train/imgs"
input_folder_masks = "../../data/foreback/processed/train/masks"
output_folder_crop_imgs = "../../data/foreback/processed/train/resize_crop"
output_folder_crop_masks = "../../data/foreback/processed/train/resize_crop_masks"
output_folder_pad_imgs = "../../data/foreback/processed/train/resize_pad"
output_folder_pad_masks = "../../data/foreback/processed/train/resize_pad_masks"
output_folder_pad_then_resize_imgs = "../../data/foreback/processed/train/resize_pad_then_resize"
output_folder_pad_then_resize_masks = "../../data/foreback/processed/train/resize_pad_then_resize_masks"

resize_and_center_crop_images(input_folder_imgs, output_folder_crop_imgs, sample_size=200)
resize_and_center_crop_images(input_folder_masks, output_folder_crop_masks, sample_size=200)
resize_and_pad_images(input_folder_imgs, output_folder_pad_imgs, sample_size=200, pad_value=0)
resize_and_pad_images(input_folder_masks, output_folder_pad_masks, sample_size=200, pad_value=0)
pad_then_resize_images(input_folder_imgs, output_folder_pad_then_resize_imgs, sample_size=200, pad_value=0)
pad_then_resize_images(input_folder_masks, output_folder_pad_then_resize_masks, sample_size=200, pad_value=0)

