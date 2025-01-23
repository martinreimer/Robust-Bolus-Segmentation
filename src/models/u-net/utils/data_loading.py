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
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

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

    @staticmethod
    def preprocess(img, scale, is_mask=False, mask_values=None):
        """
        `img` is (H, W) in float32.
        If it's a mask, we might threshold or convert values to {0,1} for binary segmentation.
        """
        h, w = img.shape
        newH, newW = int(h * scale), int(w * scale)
        assert newH > 0 and newW > 0, 'Scale too small'

        # Resize
        pil_img = Image.fromarray(img)  # shape (H, W)
        pil_img = pil_img.resize((newW, newH),
                                 resample=Image.NEAREST if is_mask else Image.BICUBIC)
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

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f"No/multiple images found for ID {name}: {img_file}"
        assert len(mask_file) == 1, f"No/multiple masks found for ID {name}: {mask_file}"

        # Load as (H, W)
        mask = load_image(mask_file[0], is_grayscale=True)
        img = load_image(img_file[0], is_grayscale=True)

        assert mask.shape == img.shape, "Image and mask have different shapes!"

        img = self.preprocess(img, self.scale, is_mask=False, mask_values=None)
        mask = self.preprocess(mask, self.scale, is_mask=True, mask_values=self.mask_values)

        # NOW add a channel dimension => (1, newH, newW)
        img_tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)

        return {
            'image': img_tensor,  # shape: [1, H, W]
            'mask': mask_tensor  # shape: [1, H, W]
        }