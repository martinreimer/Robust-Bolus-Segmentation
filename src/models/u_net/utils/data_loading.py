import logging
import numpy as np
import torch
from PIL import Image
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
import argparse
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

def load_image(filename, is_grayscale=True):
    ext = splitext(filename)[1].lower()
    if is_grayscale:
        img = Image.open(filename).convert('L')
    else:
        img = Image.open(filename)

    # Now img has shape (H, W)
    img = np.asarray(img, dtype=np.float32)
    return img


def unique_mask_values(idx, mask_dir, mask_suffix):
    """
    For scanning all masks and collecting unique values.
    """
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = load_image(mask_file, is_grayscale=True)
    if mask.ndim == 2:  # (H, W)
        return np.unique(mask)
    else:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")


class BasicDataset(Dataset):
    def __init__(self, base_dir: str, subset: str, transform=None, mask_suffix: str = ''):
        self.base_dir = Path(base_dir) / subset  # train/val/test subfolder
        self.images_dir = self.base_dir / 'imgs'
        self.mask_dir = self.base_dir / 'masks'
        self.mask_suffix = mask_suffix
        self.transform = transform

        self.ids = [
            splitext(file)[0]
            for file in listdir(self.images_dir)
            if isfile(join(self.images_dir, file)) and not file.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}')

        # lists with exact file paths
        self.img_paths = []
        self.mask_paths = []
        for file in listdir(self.images_dir):
            # Check valid extension, skip hidden etc.
            name = splitext(file)[0]
            self.img_paths.append(self.images_dir / file)
            self.mask_paths.append(self.mask_dir / f"{name}{self.mask_suffix}.png")  # or .jpg etc.

        logging.info(f'Creating {subset} dataset with {len(self.ids)} examples')

        # Collect unique mask values across all images
        with Pool() as p:
            unique_vals = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
        self.mask_values = list(sorted(np.unique(np.concatenate(unique_vals))))
        logging.info(f"Unique mask values: {self.mask_values}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        mask = load_image(self.mask_paths[idx], is_grayscale=True)
        img = load_image(self.img_paths[idx], is_grayscale=True)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            replay = augmented.get('replay', None)
            print(f"Replay info for sample {idx}: {replay}")

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True, mask_values=self.mask_values)

        img_tensor = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return {'image': img_tensor.float(), 'mask': mask_tensor.float()}


    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        img_path = self.img_paths[idx]
        mask = load_image(mask_path, is_grayscale=True)  # shape (H, W)
        img = load_image(img_path, is_grayscale=True)

        # apply preprocess -> normalize
        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True, mask_values=self.mask_values)

        # apply augmentations
        if self.transform is not None:
            # Apply augmentations
            img, mask = self.transform(image=img, mask=mask)

        # make sure the values are between 0 and 1 for mask and image
        if img.max() > 1 or img.max() < 0.1:
            raise ValueError(f"Image max value is {img.max()} and mask min value is {mask.min()}")

        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return {'image': img_tensor.float(), 'mask': mask_tensor.float()}



    @staticmethod
    def preprocess(img, is_mask=False, mask_values=None):
        """
        Preprocess images and masks: Normalize
        """
        h, w = img.shape
        # Resize
        pil_img = Image.fromarray(img)  # shape (H, W)
        img = np.asarray(pil_img, dtype=np.float32)  # shape (newH, newW)

        if is_mask:
            # Example: if masks are 0/255, convert to 0/1
            if len(mask_values) == 2 and 0 in mask_values and 255 in mask_values:
                img = (img > 127).astype(np.float32)
        else:
            # For grayscale images, normalize to [0,1]
            img = img / 255.0

        return img


# test it
# dataset = BasicDataset(base_dir='D:/Martin/thesis/data/processed/dataset_0228_final', subset='train', mask_suffix='_bolus', transform=None)
# sample = dataset[0]



def test_it():
    dataset_path = 'D:/Martin/thesis/data/processed/dataset_0228_final'
    subset='train'
    mask_suffix = '_bolus'
    num_samples=5
    batch_size=1

    # Initialize the dataset (this uses your provided BasicDataset implementation)
    dataset = BasicDataset(base_dir=dataset_path, subset=subset, mask_suffix=mask_suffix, transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logging.info(f"Dataset size: {len(dataset)}")

    # Iterate through the DataLoader and plot images with their corresponding masks.
    for idx, sample in enumerate(dataloader):
        # sample is a dictionary with 'image' and 'mask' tensors of shape (B, 1, H, W)
        image = sample['image'][0].squeeze(0).numpy()  # (H, W)
        mask = sample['mask'][0].squeeze(0).numpy()  # (H, W)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.suptitle(f"Sample {idx}")
        plt.show()

        if idx + 1 >= num_samples:
            break


if __name__ == '__main__':
    test_it()
