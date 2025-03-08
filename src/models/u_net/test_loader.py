import argparse
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test DataLoader by plotting images and masks.")
    parser.add_argument('--dataset-path', '-d', type=str, default='D:/Martin/thesis/data/processed/dataset_0228_final', help='Path to the dataset.')
    parser.add_argument('--subset', type=str, default='train', help='Subset to use: train, val, or test')
    parser.add_argument('--mask-suffix', type=str, default='_bolus', help='Suffix for mask files.')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to plot.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for DataLoader.')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Initialize the dataset (this uses your provided BasicDataset implementation)
    dataset = BasicDataset(base_dir=args.dataset_path, subset=args.subset, mask_suffix=args.mask_suffix, transform=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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

        if idx + 1 >= args.num_samples:
            break


if __name__ == '__main__':
    main()
