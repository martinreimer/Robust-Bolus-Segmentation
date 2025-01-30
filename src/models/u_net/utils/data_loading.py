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


def load_image(filename, is_grayscale=True):
    ext = splitext(filename)[1].lower()
    if ext == '.npy':
        img = np.load(filename)
    elif ext in ['.pt', '.pth']:
        img = torch.load(filename).numpy()
    else:
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
        mask_path = self.mask_paths[idx]
        img_path = self.img_paths[idx]
        mask = load_image(mask_path, is_grayscale=True)  # shape (H, W)
        img = load_image(img_path, is_grayscale=True)

        # If self.transform is Albumentations, you can pass them as (H, W) with 'mask'
        # But note: Albumentations expects (H, W, C). So you'd do:
        if self.transform is not None:
            augmented = self.transform(
                image=img[..., None],  # shape (H,W,1)
                mask=mask[..., None]
            )
            img = augmented['image'].squeeze(-1)  # back to (H, W)
            mask = augmented['mask'].squeeze(-1)

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True, mask_values=self.mask_values)

        # Verify masks are binary
        if len(self.mask_values) == 2:
            unique_vals = np.unique(mask)
            #assert np.array_equal(unique_vals,
            #[0., 1.]), f"Mask values after preprocessing are {unique_vals}, expected [0., 1.]"

        # Now shape is (H, W).
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return {
            'image': img_tensor.float(),
            'mask': mask_tensor.float()
        }
        '''
            img = self.preprocess(img, is_mask=False)
            mask = self.preprocess(mask, is_mask=True, mask_values=self.mask_values)
    
            img = np.array(img, dtype=np.float32, copy=True)
            img_tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
    
            mask = np.array(mask, dtype=np.float32, copy=True)  # Ensure it's writable
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
    
            return {
                'image': img_tensor,  # shape: [1, H, W]
                'mask': mask_tensor  # shape: [1, H, W]
            }
        '''


    @staticmethod
    def preprocess(img, is_mask=False, mask_values=None):
        """
        `img` is (H, W) in float32.
        If it's a mask, we might threshold or convert values to {0,1} for binary segmentation.
        """
        h, w = img.shape
        # Resize
        pil_img = Image.fromarray(img)  # shape (H, W)
        img = np.asarray(pil_img, dtype=np.float32)  # shape (newH, newW)

        if is_mask:
            # Example: if masks are 0/255, convert to 0/1
            if len(mask_values) == 2 and 0 in mask_values and 255 in mask_values:
                img = (img > 127).astype(np.float32)
            # shape is (newH, newW), single channel will be added below
        else:
            # For grayscale images, normalize to [0,1]
            img = img / 255.0

        return img