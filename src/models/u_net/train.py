import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
from contextlib import contextmanager

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss


@contextmanager
def wandb_run(project: str, config: dict):
    """
    Context manager for managing WandB runs.

    Args:
        project (str): The name of the WandB project.
        config (dict): Configuration dictionary to log.

    Yields:
        wandb.Run: The initialized WandB run object.
    """
    experiment = wandb.init(project=project, config=config, resume='allow', anonymous='must')
    try:
        yield experiment
    finally:
        wandb.finish()


def setup_logging(log_level: str = 'INFO', log_file: str = 'training.log') -> None:
    """
    Set up logging configuration.

    Args:
        log_level (str): Logging level as a string.
        log_file (str): File path for logging output.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_device() -> torch.device:
    """
    Determine the device to use for computations.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device: {device}')
    return device


def get_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    # Training parameters
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')  # always true
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    # Dataset options
    parser.add_argument('--train-samples', '-ts', type=int, default=None,
                        help='Maximum number of training samples to use. If not set, use all.')
    parser.add_argument('--val-samples', '-vs', type=int, default=None,
                        help='Maximum number of validation samples to use. If not set, use all.')
    parser.add_argument('--dataset-path', '-d', type=str, default='.',
                        help='Path to the dataset. Defaults to current directory.')

    # Checkpoint options
    parser.add_argument('--no-save-checkpoint', action='store_false', dest='save_checkpoint',
                        help='Do not save checkpoints after each epoch')

    args = parser.parse_args()

    # Validate arguments
    if args.train_samples is not None and args.train_samples <= 0:
        parser.error("--train-samples must be a positive integer.")
    if args.val_samples is not None and args.val_samples <= 0:
        parser.error("--val-samples must be a positive integer.")
    if not (0 <= args.val <= 100):
        parser.error("--validation must be between 0 and 100.")

    return args


def get_augmentations() -> A.Compose:
    """
    Define data augmentations for the training dataset.

    Returns:
        A.Compose: Composed augmentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5, value=0, mask_value=0),
        A.RandomBrightnessContrast(p=0.1, brightness_limit=0.025, contrast_limit=0.025),
    ], additional_targets={'mask': 'mask'})


def prepare_datasets(args: argparse.Namespace) -> Tuple[BasicDataset, BasicDataset]:
    """
    Prepare training and validation datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple[BasicDataset, BasicDataset]: Training and validation datasets.
    """
    augmentations = get_augmentations()

    train_set = BasicDataset(
        base_dir=args.dataset_path,
        subset='train',
        mask_suffix='_bolus',
        transform=augmentations
    )
    val_set = BasicDataset(
        base_dir=args.dataset_path,
        subset='val',
        mask_suffix='_bolus',
        transform=None
    )

    # Limit the number of samples if specified
    if args.train_samples is not None and args.train_samples < len(train_set):
        train_set = Subset(train_set, range(args.train_samples))
        logging.info(f"Training samples limited to: {args.train_samples}")
    if args.val_samples is not None and args.val_samples < len(val_set):
        val_set = Subset(val_set, range(args.val_samples))
        logging.info(f"Validation samples limited to: {args.val_samples}")

    logging.info(f"Total training samples: {len(train_set)}")
    logging.info(f"Total validation samples: {len(val_set)}")

    return train_set, val_set


def create_dataloaders(train_set: BasicDataset, val_set: BasicDataset, batch_size: int) -> Tuple[
    DataLoader, DataLoader]:
    """
    Create DataLoader instances for training and validation.

    Args:
        train_set (BasicDataset): Training dataset.
        val_set (BasicDataset): Validation dataset.
        batch_size (int): Batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    loader_args = dict(batch_size=batch_size, num_workers=6, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """
    Initialize the UNet model and load pretrained weights if provided.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The UNet model.
    """
    model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last).to(device=device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Load weights if requested
    if args.load:
        load_model_weights(model, args.load, device)

    return model


def load_model_weights(model: nn.Module, path: str, device: torch.device) -> None:
    """
    Load model weights from a checkpoint file.

    Args:
        model (nn.Module): The model to load weights into.
        path (str): Path to the checkpoint file.
        device (torch.device): Device to map the weights to.
    """
    if not os.path.exists(path):
        logging.error(f'Checkpoint file {path} does not exist.')
        raise FileNotFoundError(f'Checkpoint file {path} not found.')
    state_dict = torch.load(path, map_location=device)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {path}')


def log_train_augment_preview(dataset: BasicDataset, fixed_indices: list, epoch: int, experiment,
                              count: int = 3) -> None:
    """
    Log augmented samples to WandB for visualization.

    Args:
        dataset (BasicDataset): The dataset to sample from.
        fixed_indices (list): List of indices to log.
        epoch (int): Current epoch number.
        experiment: WandB run instance.
        count (int, optional): Number of samples to log. Defaults to 3.
    """
    for idx in fixed_indices[:count]:
        try:
            sample = dataset[idx]
            img_np = sample['image'].cpu().numpy()
            mask_np = sample['mask'].cpu().numpy()

            experiment.log({
                f"train_augments/sample_{idx}/epoch_{epoch}": [
                    wandb.Image(img_np, caption=f"Sample {idx} - Augmented Input"),
                    wandb.Image(mask_np, caption=f"Sample {idx} - Augmented Mask"),
                ]
            })
        except IndexError:
            logging.warning(f"Index {idx} is out of range for the dataset.")


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        amp: bool,
        grad_scaler: torch.cuda.amp.GradScaler,
        experiment,
        global_step: int,
        epoch: int
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for computations.
        amp (bool): Whether to use Automatic Mixed Precision.
        grad_scaler (torch.cuda.amp.GradScaler): Gradient scaler for AMP.
        experiment: WandB run instance.
        global_step (int): Current global step count.
        epoch (int): Current epoch number.

    Returns:
        Tuple[float, float]: Average loss and gradient norm for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    epoch_grad_norm = 0.0
    num_batches = 0

    with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch in loader:
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.amp.autocast(device_type=device.type, enabled=amp):
                masks_pred = model(images)
                loss_bce = criterion(masks_pred, true_masks)
                loss_dice = dice_loss(torch.sigmoid(masks_pred), true_masks, multiclass=False)
                loss = loss_bce + loss_dice

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()

            # Unscale gradients and compute gradient norm
            grad_scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_grad_norm += total_norm.item()

            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.size(0))
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

            # Log training loss per batch
            experiment.log({'train_loss': loss.item(), 'step': global_step, 'epoch': epoch})

    avg_loss = epoch_loss / num_batches
    avg_grad_norm = epoch_grad_norm / num_batches
    return avg_loss, avg_grad_norm


def log_validation_samples(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, experiment,
                           epoch: int, max_samples: int = 3) -> None:
    """
    Log sample predictions from the validation set to WandB.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device for computations.
        amp (bool): Whether to use Automatic Mixed Precision.
        experiment: WandB run instance.
        epoch (int): Current epoch number.
        max_samples (int, optional): Maximum number of samples to log. Defaults to 3.
    """
    model.eval()
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                pred_masks = model(images)
            pred_masks_bin = (torch.sigmoid(pred_masks) > 0.5).float()

            for i in range(images.size(0)):
                if sample_count >= max_samples:
                    return
                experiment.log({
                    f"examples/epoch_{epoch}_sample_{sample_count}": [
                        wandb.Image(images[i].cpu(), caption="Input"),
                        wandb.Image(true_masks[i].cpu(), caption="True Mask"),
                        wandb.Image(pred_masks_bin[i].cpu(), caption="Predicted Mask"),
                    ]
                })
                sample_count += 1


def save_final_model(model: nn.Module, args: argparse.Namespace) -> None:
    """
    Save the final trained model to the checkpoints directory.

    Args:
        model (nn.Module): The trained model.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    run_dir = Path("./runs")
    latest_run = sorted(run_dir.glob('run-*'), key=os.path.getmtime)[-1]
    checkpoints_dir = latest_run / "checkpoints"
    final_model_path = checkpoints_dir / 'model.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Final model saved to {final_model_path}')


def train_model(
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace
) -> None:
    """
    Train the UNet model.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): Device for computations.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Create a run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("./runs") / f"run-{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Checkpoints and final model will be saved to: {checkpoints_dir}")

    # Initialize WandB
    with wandb_run(project='U-Net', config=vars(args)) as experiment:
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)

        logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {args.save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {args.amp}
        ''')

        # Setup optimizer, scheduler, and loss function
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-8,
                                  momentum=0.999,
                                  foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        grad_scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
        criterion = nn.BCEWithLogitsLoss()

        global_step = 0
        fixed_indices_to_track = [0, len(train_loader.dataset) // 2, len(train_loader.dataset) - 1] if len(
            train_loader.dataset) > 0 else []

        for epoch in range(1, args.epochs + 1):
            epoch_loss, epoch_grad_norm = train_one_epoch(
                model, train_loader, optimizer, criterion, device, args.amp, grad_scaler, experiment, global_step, epoch
            )
            global_step += len(train_loader)

            # Validation
            val_dice, val_loss = evaluate(
                net=model,
                dataloader=val_loader,
                device=device,
                amp=args.amp,
                criterion=criterion
            )
            scheduler.step(val_dice)

            logging.info(f'''Epoch {epoch} / {args.epochs}:
                Train Loss: {epoch_loss:.4f}
                Val Dice:   {val_dice:.4f}
                Val Loss:   {val_loss:.4f}
                Grad Norm:  {epoch_grad_norm:.4f}
            ''')

            # Log epoch-level metrics to WandB
            experiment.log({
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'train_loss_epoch': epoch_loss,
                'val_dice_epoch': val_dice,
                'val_loss_epoch': val_loss,
                'grad_norm_epoch': epoch_grad_norm
            })

            # Log sample images from validation
            log_validation_samples(model, val_loader, device, args.amp, experiment, epoch)

            # Save checkpoint
            if args.save_checkpoint:
                ckpt_path = checkpoints_dir / f'checkpoint_epoch{epoch}.pth'
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f'Checkpoint {epoch} saved to {ckpt_path}')

            # Log training augmentations
            log_train_augment_preview(train_loader.dataset, fixed_indices_to_track, epoch, experiment)


def main():
    """
    Main function to orchestrate the training process.
    """
    args = get_args()
    setup_logging()
    device = get_device()
    train_set, val_set = prepare_datasets(args)
    train_loader, val_loader = create_dataloaders(train_set, val_set, args.batch_size)
    model = build_model(args, device)

    try:
        train_model(model, device, train_loader, val_loader, args)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! Enabling checkpointing for gradient to reduce memory usage.')
        torch.cuda.empty_cache()
        if hasattr(model, 'use_checkpointing'):
            model.use_checkpointing()  # If your model supports gradient checkpointing
            train_model(model, device, train_loader, val_loader, args)
        else:
            logging.error('Model does not support checkpointing. Exiting.')
            exit(1)


if __name__ == '__main__':
    main()
#python train.py -e 10 -b 32 -ts 200 -vs 50 -d ../../../data/processed/dataset_first_experiments
