'''
python train.py --epochs 40 -b 8 -l 1e-5 --loss combined -d D:\Martin\thesis\data\processed\dataset_0228_final

python train.py --epochs 25 -d D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504_test_final_roi_crop -b 8 -l 1e-3 --loss dice -ms _bolus --optimizer adamax --scheduler plateau --model-source smp --encoder-name resnet34 --encoder-weights imagenet --encoder-depth 5 --decoder-interpolation nearest --decoder-use-norm batchnorm
python train.py --epochs 25 -d D:\Martin\thesis\data\processed\dataset_labelbox_export_test_2504_test_final_roi_crop -b 8 -l 1e-3 --loss dice -ms _bolus --optimizer adamax --scheduler plateau



'''
from __future__ import annotations

import csv
from segmentation_models_pytorch.losses import (
    DiceLoss,
    JaccardLoss,
    TverskyLoss,
    FocalLoss,
    LovaszLoss,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss,
    MCCLoss,
)
from segmentation_models_pytorch.losses.constants import BINARY_MODE
from torchinfo import summary
import sys
import pandas as pd
import random
import numpy as np
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

import wandb
from contextlib import contextmanager

import albumentations as A
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# disable user warnings
import warnings
warnings.filterwarnings("ignore")

# use external segmentation models
import segmentation_models_pytorch as smp

# my stuff
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
#from utils.loss import dice_loss, DiceLoss, IoULoss, FocalLoss, LogCoshDiceLoss, TverskyLoss
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler

import os
os.environ["NO_COLOR"] = "1"


@contextmanager
def wandb_run(project: str, config: dict, output_dir: str):
    """
    Context manager for managing WandB runs.
    """
    experiment = wandb.init(
        project=project,
        config=config,
        resume='allow',
        anonymous='must',
        dir=output_dir
    )
    try:
        print(f"WandB {experiment.project} - {experiment.name}")
        yield experiment
    finally:
        wandb.finish(quiet=True)



def setup_logging(log_level: str = 'INFO', log_file: str = 'training.log', output_dir: str = 'D:/Martin/thesis/training_runs') -> None:
    """
    Set up logging configuration.

    Args:
        log_level (str): Logging level as a string.
        log_file (str): File path for logging output.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Construct the full path for the log file in output_dir
    log_file = Path(output_dir) / 'training.log'

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
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

    parser.add_argument('--img-size', type=int, nargs=2, metavar=('H', 'W'), default=[256, 256], help = 'Input image height and width (e.g. 256 256)')
    parser.add_argument('--num-workers', type=int, default=0, help = 'Number of DataLoader workers')
    parser.add_argument('--persistent-workers', action='store_true', default=False, help = 'Keep DataLoader workers alive between epochs')
    parser.add_argument('--seed', type=int, default=42, help = 'Random seed for torch, numpy, python.random')



    # Dataset options
    parser.add_argument('--dataset-path', '-d', type=str, help='Path to the dataset. Defaults to current directory.')
    parser.add_argument('--train-samples', '-ts', type=int, default=None,
                        help='Maximum number of training samples to use. For Quick Checks.')
    parser.add_argument('--val-samples', '-vs', type=int, default=None,
                        help='Maximum number of validation samples to use. For Quick Checks.')
    parser.add_argument('--mask-suffix', '-ms', type=str, default='_bolus',
                        help='Suffix for mask files. Defaults to "_bolus".')
    # Output directory
    parser.add_argument('--output-dir', type=str, default='D:/Martin/thesis/training_runs', help='Directory to store all outputs (runs, checkpoints, logs, etc.). Default is "D:/Martin/thesis/training_runs".')
    # Checkpoint options
    parser.add_argument('--no-save-checkpoint', action='store_false', dest='save_checkpoint',
                        help='Do not save checkpoints after each epoch')
    # Training parameters
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')  # always true
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    # Loss function options
    parser.add_argument('--loss', type=str, default='dice', choices=['bce', 'dice', 'iou', 'bce_dice', 'tversky', 'logcosh_dice', 'focal'], help='Loss function to use. Options: "bce", "dice", "iou", "bce_dice"')
    parser.add_argument('--combined-bce-weight', type=float, default=1.0, help='Weight for the BCE component in combined loss')
    parser.add_argument('--combined-dice-weight', type=float, default=1.0, help='Weight for the Dice component in combined loss')
    # ──────────────── Loss-specific hyper-parameters ────────────────
    # Focal
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal α – class-balancing factor')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal γ – focusing parameter')
    # Tversky
    parser.add_argument('--tversky-alpha', type=float, default=0.5, help='Tversky α – weight for FN')
    parser.add_argument('--tversky-beta', type=float, default=0.5, help='Tversky β – weight for FP')

    # Learning rate scheduler options:
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['step', 'exponential', 'plateau'], help='Learning rate scheduler to use: "step" for StepLR, "exponential" for ExponentialLR, "plateau" for ReduceLROnPlateau')
    parser.add_argument('--lr-step-size', type=int, default=5, help='Step size for StepLR scheduler (number of epochs between decays)')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='Decay factor for the scheduler (gamma)')
    parser.add_argument('--lr-patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr-mode', type=str, default='min', choices=['min', 'max'], help='Mode for ReduceLROnPlateau scheduler; typically "min" for loss')


    # Modelling specifications
    parser.add_argument('--model-source', type=str, default='custom', choices=['smp', 'custom'], help='Model source: "smp" for segmentation_models_pytorch, "custom" for custom implementation')
    parser.add_argument('--filters', type=str, default='64,128,256,512,1024', help='Comma-separated list of filter sizes for U-Net layers (e.g., 32,64,128,256)')
    parser.add_argument('--use-attention', action='store_true', default=False, help='Use attention gates in U-Net')

    # SMP specific options
    parser.add_argument('--smp-model', type=str, default='Unet', choices=['Unet', 'UnetPlusPlus', 'Segformer'], help='Model type in segmentation_models_pytorch. Options: "Unet", "UnetPlusPlus", "Segformer"')
    parser.add_argument('--encoder-name', type=str, default=None, help='Encoder name for segmentation_models_pytorch (f.e. resnet34)')
    parser.add_argument('--encoder-weights', type=str, default=None, help='Pretrained weights for encoder in segmentation_models_pytorch (f.e. imagenet)')
    parser.add_argument('--encoder-depth', type=int, default=5, choices=[3, 4, 5], help='Depth of the encoder in segmentation_models_pytorch (3-5)')
    parser.add_argument('--decoder-interpolation', type=str, default='nearest', choices=['nearest', 'bilinear', 'bicubic', 'area', 'nearest-exact'], help='Interpolation method for decoder in segmentation_models_pytorch')
    parser.add_argument('--decoder-use-norm', type=str, default='batchnorm', choices=[False, 'batchnorm', 'identity', 'layernorm', 'instancenorm', 'inplace'], help='Normalization type for decoder in segmentation_models_pytorch. Options: "batchnorm", "identity", "layernorm", "instancenorm", "inplace"')

    # print value for encoder-weights
    print(f"Encoder weights: {parser.get_default('encoder_weights')}")
    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad', 'adamax'],
                        help='Optimizer to use. Options: "adam", "nadam", "rmsprop", "sgd", "adadelta", "adagrad", "adamax"')
    # for SGD
    parser.add_argument('--sgd-momentum', type=float, default=0.9, help='Momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--sgd-nesterov', dest='sgd_nesterov', action='store_true', help='Enable Nesterov momentum for SGD')
    parser.add_argument('--no-sgd-nesterov', dest='sgd_nesterov', action='store_false', help='Disable Nesterov momentum for SGD')
    parser.set_defaults(sgd_nesterov=True)
    # for Adam/NAdam
    parser.add_argument('--adam-beta1', type=float, default=0.9, help='Beta1 for Adam/NAdam optimizer (default: 0.9)')
    parser.add_argument('--adam-beta2', type=float, default=0.999, help='Beta2 for Adam/NAdam optimizer (default: 0.999)')
    parser.add_argument('--adam-eps', type=float, default=1e-8, help='Epsilon for Adam/Nadam optimizer (default: 1e-8)')
    parser.add_argument('--adam-weight-decay', type=float, default=0, help='Weight decay for Adam/NAdam optimizer (default: 0)')
    # for RMSprop
    parser.add_argument('--rmsprop-momentum', type=float, default=0.9, help='Momentum for RMSprop optimizer (default: 0.9)')
    parser.add_argument('--rmsprop-weight-decay', type=float, default=1e-8, help='Weight decay for RMSprop optimizer (default: 1e-8)')


    args = parser.parse_args()

    # Validate arguments
    # check if we choose custom model, that smp specific arguments are not set
    if args.model_source == 'custom':
        if args.encoder_name is not None or args.encoder_weights is not None:
            parser.error("Encoder name and weights are only applicable for segmentation_models_pytorch models.")
    elif args.model_source == 'smp':
        if args.encoder_weights == "None":
            args.encoder_weights = None
    if args.dataset_path is None or not os.path.exists(args.dataset_path):
        parser.error("Dataset path is required and must exist.")
    if args.output_dir is None or not os.path.exists(args.output_dir):
        parser.error("Output directory is required and must exist.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer.")
    if args.lr <= 0:
        parser.error("--learning-rate must be a positive float.")
    if args.lr_step_size <= 0:
        parser.error("--lr-step-size must be a positive integer.")
    if args.lr_gamma <= 0:
        parser.error("--lr-gamma must be a positive float.")
    if args.lr_patience <= 0:
        parser.error("--lr-patience must be a positive integer.")
    if args.train_samples is not None and args.train_samples <= 0:
        parser.error("--train-samples must be a positive integer.")
    if args.val_samples is not None and args.val_samples <= 0:
        parser.error("--val-samples must be a positive integer.")
    try:
        args.filters = [int(f) for f in args.filters.split(',')]
    except ValueError:
        parser.error("Invalid filter sizes. Please provide a comma-separated list of integers (e.g., 32,64,128,256).")
    return args


# Define intensity transformations (applied only to the image)
intensity_transforms = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), p=0.25)
    ], p=0.3)
])
# Spatial transforms (applied to both image and mask)
spatial_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=5, p=0.5, border_mode=0),  # using a lower rotation limit
    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(-0.05, 0.05),  # lower translation limits
        shear=(-3, 3),                  # lower shear
        p=0.5,
        border_mode=0                   # consider using cv2.BORDER_REFLECT_101 if desired
    )
], additional_targets={'mask': 'mask'})




# Define the augmentation function at the module level
def augment_fn(image, mask):
    # Apply spatial transformations (both image and mask)
    augmented = spatial_transforms(image=image[..., None], mask=mask[..., None])
    image_aug = augmented['image'].squeeze(-1)
    mask_aug = augmented['mask'].squeeze(-1)
    # Apply intensity transformations only to the image
    image_aug = intensity_transforms(image=image_aug[..., None])['image'].squeeze(-1)
    return image_aug, mask_aug

def get_augmentations():
    """
    Returns the augmentation function.
    """
    return augment_fn


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
        mask_suffix=args.mask_suffix,
        transform=augmentations
    )
    val_set = BasicDataset(
        base_dir=args.dataset_path,
        subset='val',
        mask_suffix=args.mask_suffix,
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



def create_dataloaders(train_set: BasicDataset, val_set: BasicDataset, batch_size: int, num_workers: int, persistent_workers: bool) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader instances for training and validation.

    Args:
        train_set (BasicDataset): Training dataset.
        val_set (BasicDataset): Validation dataset.
        batch_size (int): Batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True, persistent_workers = False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)#, prefetch_factor=2)
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
    if args.model_source == 'smp':
        # U-Net
        if args.smp_model == 'Unet':
            model = smp.Unet(
                encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=1, classes=1,
                decoder_attention_type=None, activation=None, encoder_depth=args.encoder_depth, decoder_use_batchnorm=True,
            )
        elif args.smp_model == 'UNetPlusPlus':
            model = smp.UnetPlusPlus(
                encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=1, classes=1,
                decoder_attention_type=None, activation=None, encoder_depth=args.encoder_depth, decoder_use_batchnorm=True,
            )
        elif args.smp_model == 'Segformer':
            model = smp.Segformer(
                encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=1, classes=1,
                decoder_attention_type=None, activation=None, encoder_depth=args.encoder_depth, decoder_use_batchnorm=True,
            )
        else:
            raise ValueError(f"Unsupported SMP model: {args.smp_model}")

    elif args.model_source == 'custom':
        model = UNet(n_channels=1, n_classes=1, filters=args.filters, bilinear=args.bilinear, use_attention=args.use_attention)
    else:
        raise ValueError(f"Unsupported model source: {args.model_source}")
    model = model.to(memory_format=torch.channels_last).to(device=device)
    return model

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


def compute_imbalance_pos_weight(dataset: BasicDataset) -> float:
    """
    Compute the positive class weight for BCE loss based on the inverse class frequency
    from the training dataset masks.

    Args:
        dataset (BasicDataset): The training dataset (or its subset).

    Returns:
        float: The computed pos_weight = total_negatives / total_positives.
    """
    total_pos = 0.0
    total_pixels = 0.0
    for sample in dataset:
        # Assuming mask is a tensor or can be converted to one
        mask = sample['mask']
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()
        total_pos += mask.sum().item()
        total_pixels += mask.numel()
    total_neg = total_pixels - total_pos
    # Avoid division by zero
    pos_weight = total_neg / (total_pos + 1e-8)
    logging.info(f'Computed pos_weight: {pos_weight:.4f} (Total negatives: {total_neg}, Total positives: {total_pos})')
    return pos_weight


def get_loss(
    loss_name: str,
    *,
    pos_weight: float | None = None,
    combined_bce_weight: float = 1.0,
    combined_dice_weight: float = 1.0,
    focal_alpha: float | None = None,
    focal_gamma: float | None = None,
    tversky_alpha: float | None = None,
    tversky_beta: float | None = None,
) -> nn.Module | callable:
    """
    Return an SMP loss by name. Supported loss_name values:
      'bce', 'dice', 'iou', 'bce_dice', 'tversky', 'focal',
      'lovasz', 'soft_crossentropy', 'mcc'
    """
    name = loss_name.lower()
    mode = BINARY_MODE

    # ----- BCE -----
    if name == 'bce':
        if pos_weight is not None:
            return SoftBCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        return SoftBCEWithLogitsLoss()

    # ----- Dice -----
    if name == 'dice':
        return DiceLoss(mode=mode, from_logits=True)

    # ----- IoU / Jaccard -----
    if name in ('iou', 'jaccard'):
        return JaccardLoss(mode=mode, from_logits=True)

    # ----- BCE + Dice -----
    if name in ('bce_dice', 'combined'):
        def _bce_dice(pred, target):
            bce = SoftBCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight) if pos_weight is not None else None
            )(pred, target)
            dice = DiceLoss(mode=mode, from_logits=True)(pred, target)
            return combined_bce_weight * bce + combined_dice_weight * dice
        return _bce_dice

    # ----- Tversky -----
    if name == 'tversky':
        alpha = tversky_alpha or combined_bce_weight
        beta  = tversky_beta  or combined_dice_weight
        return TverskyLoss(mode=mode, alpha=alpha, beta=beta, from_logits=True)

    # ----- Focal -----
    if name == 'focal':
        alpha = focal_alpha if focal_alpha is not None else combined_bce_weight
        gamma = focal_gamma if focal_gamma is not None else combined_dice_weight
        return FocalLoss(mode=mode, alpha=alpha, gamma=gamma)

    # ----- Lovasz -----
    if name == 'lovasz':
        return LovaszLoss(mode=mode, from_logits=True)

    # ----- Soft Cross-Entropy (mainly for multiclass) -----
    if name == 'soft_crossentropy':
        return SoftCrossEntropyLoss()

    # ----- MCC -----
    if name == 'mcc':
        return MCCLoss()

    raise ValueError(f"Unsupported loss: {loss_name}")

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion,
        device: torch.device,
        amp: bool,
        grad_scaler: torch.cuda.amp.GradScaler,
        experiment,
        global_step: int,
        epoch: int,
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
    stats = {
        'data_load_time': 0.0,
        'forward_time': 0.0,
        'backward_time': 0.0,
        'step_time': 0.0,
        'total_loss': 0.0,
        'num_batches': 0
    }

    data_timer, fwd_timer, bwd_timer, step_timer = 0, 0, 0, 0
    batch_start = time.time()

    for batch in loader:
        # 1) data-loading
        t0 = time.time()
        images, true_masks = batch['image'], batch['mask']
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        stats['data_load_time'] += time.time() - t0

        # 2) forward + loss
        t1 = time.time()
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks)
        stats['forward_time'] += time.time() - t1

        # 3) backward
        t2 = time.time()
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        stats['backward_time'] += time.time() - t2

        # 4) step & scaler update
        t3 = time.time()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        stats['step_time'] += time.time() - t3

        stats['total_loss'] += loss.item()
        stats['num_batches'] += 1
        global_step += 1

    # average loss
    stats['avg_loss'] = stats['total_loss'] / max(stats['num_batches'], 1)
    stats['train_epoch_time'] = (time.time() - batch_start) / 60.0
    return stats




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
                    f"examples/sample_{sample_count}_epoch_{epoch}": [
                        wandb.Image(images[i].cpu(), caption="Input"),
                        wandb.Image(true_masks[i].cpu(), caption="True Mask"),
                        wandb.Image(pred_masks_bin[i].cpu(), caption="Predicted Mask"),
                    ]
                })
                sample_count += 1


def dump_experiment_info(
        model: torch.nn.Module,
        run_dir: Path,
        args: argparse.Namespace,
        in_channels: int = 1,
        classes: int = 1
) -> None:
    """
    Dump model architecture and experiment settings to files in run_dir.

    Creates:
      - experiment_info.txt
      - model_summary.txt
      - layer_params.csv
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Gather basic info
    cli_line = " ".join(sys.argv)
    H, W = args.img_size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2) Write experiment_info.txt
    info_txt = run_dir / "experiment_info.txt"
    with open(info_txt, "w") as f:
        f.write("=== Experiment Configuration ===\n")
        f.write(f"CLI: {cli_line}\n")
        f.write(f"Image size  : {H}×{W}\n")
        f.write(f"Num workers : {args.num_workers}\n")
        f.write(f"Persistent workers: {args.persistent_workers}\n")
        f.write(f"Seed        : {args.seed}\n\n")
        f.write("=== Model Parameters ===\n")
        f.write(f"in_channels      : {in_channels}\n")
        f.write(f"classes          : {classes}\n")
        f.write(f"total parameters : {total_params:,}\n")
        f.write(f"trainable params : {trainable_params:,}\n")

    # 3) Dump torchinfo summary
    summary_txt = run_dir / "model_summary.txt"
    model_info = summary(
        model,
        input_size=(1, in_channels, H, W),
        verbose=0
    )

    # write with UTF-8 so box-drawing and other unicode chars aren’t rejected
    with open(summary_txt, "w", encoding="utf-8", errors="replace") as f:
        f.write(str(model_info))

    # 4) Layer-wise parameter counts
    layers = [(name, param.numel()) for name, param in model.named_parameters()]
    df = pd.DataFrame(layers, columns=["layer_name", "param_count"])
    df.to_csv(run_dir / "layer_params.csv", index=False)


def train_model(
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pos_weight = compute_imbalance_pos_weight(train_loader.dataset)
    # keep track of the best val_loss & epoch
    best_val_loss = float('inf')
    best_epoch    = 0

    with wandb_run(project='U-Net', config=vars(args), output_dir=args.output_dir) as experiment:
        #set up
        project_name = experiment.project
        run_name = experiment.name
        cli_line = " ".join(sys.argv)
        run_dir = Path(args.output_dir) / project_name / "runs" / f"{run_name}"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints and final model will be saved to: {checkpoints_dir}")
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
        dump_experiment_info(model = model, run_dir = run_dir, args = args, in_channels = 1, classes = 1)
        # ── Optimizer, scheduler, and loss function ─────────────────────────────
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        criterion = get_loss(
            loss_name=args.loss,
            pos_weight=pos_weight,
            combined_bce_weight=args.combined_bce_weight,
            combined_dice_weight=args.combined_dice_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
        )

        global_step = 0
        fixed_indices_to_track = [0, len(train_loader.dataset) // 2, len(train_loader.dataset) - 1] if len(train_loader.dataset) > 0 else []

        total_training_start = time.time()
        for epoch in range(1, args.epochs + 1):
            start_epoch_time = time.time()
            # --- training ---
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, device, args.amp, grad_scaler, experiment, global_step, epoch
            )
            epoch_loss, epoch_grad_norm = train_stats["avg_loss"], 0#train_stats["grad_norm"]
            global_step += len(train_loader)

            # Validation and timing
            val_start = time.time()
            # ── validation ──────────────────────────────────────────────────────────
            val_metrics = evaluate(net=model,
                                   dataloader=val_loader,
                                   device=device,
                                   amp=args.amp,
                                   criterion=criterion,
                                   pos_weight=pos_weight)
            # Print out the detailed timings
            val_epoch_time = val_metrics['val_time']
            total_epoch_time = (time.time() - start_epoch_time) / 60.0

            logging.info(
                f"""Epoch {epoch}/{args.epochs}
            ─────────────────────────────────────────────────────────
              Train Loss        : {epoch_loss:.4f}
              ─ Validation
                • Loss          : {val_metrics['val_loss']:.4f}
                • Dice          : {val_metrics['val_dice']:.4f}
                • IoU           : {val_metrics['val_iou']:.4f}
              Grad-norm         : {epoch_grad_norm:.4f}
              Time (epoch)      : {total_epoch_time:.2f} min
                · train         : {train_stats['train_epoch_time']:.2f} min
                  · fwd           : {train_stats['forward_time']:.2f} s
                  · bwd           : {train_stats['backward_time']:.2f} s
                  · step          : {train_stats['step_time']:.2f} s
                · val total     : {val_metrics['val_time']:.2f} min
                  · data load  : {val_metrics['val_time_data_load']:.2f} s
                  · forward    : {val_metrics['val_time_forward']:.2f} s
                  · metrics    : {val_metrics['val_time_metrics']:.2f} s
            ─────────────────────────────────────────────────────────"""
            )

            # ── WandB / CSV logging ────────────────────────────────────────────────
            experiment.log({
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'train_loss_epoch': epoch_loss,
                'grad_norm_epoch': epoch_grad_norm,
                'train_time_min': train_stats['train_epoch_time'],
                'val_time_min': val_epoch_time,
                'total_epoch_time_min': total_epoch_time,
                # ▸ dump every val_* metric returned
                **val_metrics
            })

            # keep track of the best
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_epoch = epoch

            # ── Weight & Gradient Histograms ────────────────────────────────────
            for name, param in model.named_parameters():
                # weights
                experiment.log({
                    f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy()),
                    "epoch": epoch
                })
                # gradients (if they exist)
                if param.grad is not None:
                    experiment.log({
                        f"grads/{name}": wandb.Histogram(param.grad.detach().cpu().numpy()),
                        "epoch": epoch
                    })

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

            try:
                log_validation_samples(model, val_loader, device, args.amp, experiment, epoch)
            except Exception as e:
                logging.error(f"Error logging validation samples: {e}")

            if args.save_checkpoint and epoch % 2 == 0:
                ckpt_path = checkpoints_dir / f'checkpoint_epoch{epoch}.pth'


                if args.model_source == 'custom':
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'state_dict': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                        'grad_scaler_state': grad_scaler.state_dict() if grad_scaler is not None else None,
                        'config': {
                            'model_source': args.model_source,
                            'n_channels': 1,
                            'n_classes': 1,
                            'filters': args.filters,
                            'bilinear': args.bilinear,
                            'use_attention': args.use_attention,
                        },
                        'mask_values': [0, 1],
                    }, ckpt_path)
                elif args.model_source == 'smp':

                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'state_dict': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                        'grad_scaler_state': grad_scaler.state_dict() if grad_scaler is not None else None,
                        'config': {
                            'model_source': args.model_source,
                            'smp_model': args.smp_model,
                            'encoder_name': args.encoder_name,
                            'encoder_weights': args.encoder_weights,
                            'in_channels': 1,
                            'classes': 1,
                            'decoder_attention_type': None,
                            'activation': None,
                            'encoder_depth': args.encoder_depth,
                            'decoder_interpolation': args.decoder_interpolation,
                            'decoder_use_norm': args.decoder_use_norm,
                        },
                        'mask_values': [0, 1],
                    }, ckpt_path)
                #else: print(f"Checkpoint not saved, model source is not supported")
                logging.info(f'Checkpoint {epoch} saved to {ckpt_path}')
            try:
                pass#log_train_augment_preview(train_loader.dataset, fixed_indices_to_track, epoch, experiment)
            except Exception as e:
                logging.error(f"Error logging training augmentations: {e}")

        total_training_time = (time.time() - total_training_start) / 60.0
        experiment.log({'total_training_time_min': total_training_time})
        logging.info(f"Total training time: {total_training_time:.2f} minutes")

        # once all epochs are done, dump to CSV
        summary_path = Path(args.output_dir) / "experiments_summary.csv"
        header = not summary_path.exists()
        with open(summary_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            if header:
                writer.writerow(['run_name', 'best_epoch', 'best_val_loss', 'cli', 'training_time', 'timestamp'])
            writer.writerow([run_name, best_epoch, best_val_loss, cli_line, total_training_time, timestamp])


def main():
    """
    Main function to orchestrate the training process.
    """
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    setup_logging(output_dir=args.output_dir)
    device = get_device()
    print(f"aaa-{args.model_source}-")

    train_set, val_set = prepare_datasets(args)
    train_loader, val_loader = create_dataloaders(train_set, val_set, batch_size=args.batch_size,num_workers = args.num_workers,persistent_workers = args.persistent_workers)

    model = build_model(args, device)
    #print(f"Model:\n{model}")
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