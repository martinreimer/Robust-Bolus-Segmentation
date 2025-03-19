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

from tqdm import tqdm
import wandb
from contextlib import contextmanager

# my stuff
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler

import albumentations as A
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# disable user warnings
import warnings
warnings.filterwarnings("ignore")


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
        wandb.finish()



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

    # Training parameters
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')  # always true
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    # Loss function options
    parser.add_argument('--loss', type=str, default='dice', choices=['bce', 'dice', 'iou', 'combined'],
                        help='Loss function to use. Options: "bce", "dice", "iou", "combined"')
    parser.add_argument('--combined-bce-weight', type=float, default=1.0,
                        help='Weight for the BCE component in combined loss')
    parser.add_argument('--combined-dice-weight', type=float, default=1.0,
                        help='Weight for the Dice component in combined loss')

    # Learning rate scheduler options:
    # Scheduler options
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['step', 'exponential', 'plateau'],
                        help='Learning rate scheduler to use: "step" for StepLR, "exponential" for ExponentialLR, '
                             '"plateau" for ReduceLROnPlateau')
    parser.add_argument('--lr-step-size', type=int, default=5,
                        help='Step size for StepLR scheduler (number of epochs between decays)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Decay factor for the scheduler (gamma)')
    parser.add_argument('--lr-patience', type=int, default=3,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr-mode', type=str, default='min', choices=['min', 'max'],
                        help='Mode for ReduceLROnPlateau scheduler; typically "min" for loss')

    # unet stuff
    parser.add_argument('--filters', type=str, default='64,128,256,512,1024',
                        help='Comma-separated list of filter sizes for U-Net layers (e.g., 32,64,128,256)')
    # Dataset options
    parser.add_argument('--train-samples', '-ts', type=int, default=None,
                        help='Maximum number of training samples to use. If not set, use all.')
    parser.add_argument('--val-samples', '-vs', type=int, default=None,
                        help='Maximum number of validation samples to use. If not set, use all.')
    parser.add_argument('--dataset-path', '-d', type=str, default='.',
                        help='Path to the dataset. Defaults to current directory.')
    parser.add_argument('--mask-suffix', '-ms', type=str, default='_bolus',
                        help='Suffix for mask files. Defaults to "_bolus".')

    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        choices=['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad', 'adamax'],
                        help='Optimizer to use. Options: "adam", "nadam", "rmsprop", "sgd", "adadelta", "adagrad", "adamax"')

    # Checkpoint options
    parser.add_argument('--no-save-checkpoint', action='store_false', dest='save_checkpoint',
                        help='Do not save checkpoints after each epoch')

    # Output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default='D:/Martin/thesis/training_runs',
        help='Directory to store all outputs (runs, checkpoints, logs, etc.). '
             'Default is "D:/Martin/thesis/training_runs".'
    )
    args = parser.parse_args()

    # Validate arguments
    if args.train_samples is not None and args.train_samples <= 0:
        parser.error("--train-samples must be a positive integer.")
    if args.val_samples is not None and args.val_samples <= 0:
        parser.error("--val-samples must be a positive integer.")
    if not (0 <= args.val <= 100):
        parser.error("--validation must be between 0 and 100.")
        # Convert the filters string to a list of integers
    try:
        args.filters = [int(f) for f in args.filters.split(',')]
    except ValueError:
        parser.error(
            "Invalid filter sizes. Please provide a comma-separated list of integers (e.g., 32,64,128,256)."
        )
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
    model = UNet(n_channels=1, n_classes=1, filters=args.filters, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last).to(device=device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling\n'
                 f'\tFilter sizes: {model.filters}')

    # Load weights if requested (unchanged)
    if args.load:
        load_model_weights(model, args.load, device)

    return model

'''
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
'''

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



def get_loss(loss_name: str, pos_weight: float = None,
             combined_bce_weight: float = 1.0, combined_dice_weight: float = 1.0):
    """
    Returns a loss function based on the provided loss_name.

    Args:
        loss_name (str): Choice of loss ("bce", "dice", "iou", "combined").
        pos_weight (float, optional): Positive class weight for BCE loss (based on inverse class frequency).
        combined_bce_weight (float): Weight for the BCE component in the combined loss.
        combined_dice_weight (float): Weight for the Dice component in the combined loss.

    Returns:
        A callable loss function.
    """
    loss_name = loss_name.lower()

    if loss_name == 'bce':
        # Use pos_weight if provided
        if pos_weight is not None:
            print(f"Used Loss: BCE with pos_weight={pos_weight}")
            pos_weight_tensor = torch.tensor([pos_weight], device='cuda')
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            print("Used Loss: BCE")
            return nn.BCEWithLogitsLoss()

    elif loss_name == 'dice':
        # Dice loss callable
        print("Used Loss: Dice")
        return lambda pred, target: dice_loss(torch.sigmoid(pred), target, multiclass=False)

    elif loss_name == 'iou':
        print("Used Loss: IoU")
        # IoU loss implementation
        def iou_loss(pred, target, smooth=1e-6):
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = (intersection + smooth) / (union + smooth)
            return 1 - iou

        return iou_loss

    elif loss_name == 'combined':
        # Combined loss: weighted BCE (with optional pos_weight) plus Dice loss.
        # Capture pos_weight explicitly as a default parameter.
        print(f"Used Loss: Combined (BCE weight: {combined_bce_weight}, Dice weight: {combined_dice_weight}, pos_weight: {pos_weight})")
        def combined_loss(pred, target, pos_weight=pos_weight):
            if pos_weight is not None:
                pos_weight_tensor = torch.tensor([pos_weight], device=pred.device)
                bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                bce_loss = nn.BCEWithLogitsLoss()
            loss_bce = bce_loss(pred, target)
            loss_dice = dice_loss(torch.sigmoid(pred), target, multiclass=False)
            return loss_bce + loss_dice
        return combined_loss

    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

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
                loss = criterion(masks_pred, true_masks)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()

            # Unscale gradients and compute gradient norm
            grad_scaler.unscale_(optimizer)
            #### !!!!!!!!!! i changed max norm from 1 to 10 temp
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
                    f"examples/sample_{sample_count}_epoch_{epoch}": [
                        wandb.Image(images[i].cpu(), caption="Input"),
                        wandb.Image(true_masks[i].cpu(), caption="True Mask"),
                        wandb.Image(pred_masks_bin[i].cpu(), caption="Predicted Mask"),
                    ]
                })
                sample_count += 1

'''
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
'''



def train_model(
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pos_weight = compute_imbalance_pos_weight(train_loader.dataset)

    with wandb_run(project='U-Net', config=vars(args), output_dir=args.output_dir) as experiment:
        #set up
        project_name = experiment.project
        run_name = experiment.name
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

        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        criterion = get_loss(
            loss_name=args.loss,
            pos_weight=pos_weight,
            combined_bce_weight=args.combined_bce_weight,
            combined_dice_weight=args.combined_dice_weight
        )

        global_step = 0
        fixed_indices_to_track = [0, len(train_loader.dataset) // 2, len(train_loader.dataset) - 1] if len(train_loader.dataset) > 0 else []

        total_training_start = time.time()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            epoch_loss, epoch_grad_norm = train_one_epoch(
                model, train_loader, optimizer, criterion, device, args.amp, grad_scaler, experiment, global_step, epoch
            )
            train_epoch_time = (time.time() - epoch_start) / 60.0
            global_step += len(train_loader)

            # Validation and timing
            val_start = time.time()
            val_metrics = evaluate(net=model, dataloader=val_loader, device=device, amp=args.amp, criterion=criterion, pos_weight=pos_weight)
            val_epoch_time = val_metrics['val_time']
            total_epoch_time = (time.time() - epoch_start) / 60.0

            logging.info(f'''Epoch {epoch} / {args.epochs}:
                Train Loss: {epoch_loss:.4f}
                Val Loss:   {val_metrics['val_loss']:.4f}
                Weighted CE: {val_metrics['val_weighted_ce']:.4f}
                Dice:       {val_metrics['val_dice']:.4f}
                BCE + Dice: {val_metrics['val_bce_dice']:.4f}
                IoU:        {val_metrics['val_iou']:.4f}
                Grad Norm:  {epoch_grad_norm:.4f}
                Total Epoch Time (min): {total_epoch_time:.2f}
                Train Time (min): {train_epoch_time:.2f}
                Val Time (min):   {val_epoch_time:.2f}
            ''')

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

            experiment.log({
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'train_loss': epoch_loss,
                'train_loss_epoch': epoch_loss,
                'val_loss_epoch': val_metrics['val_loss'],
                'val_weighted_ce': val_metrics['val_weighted_ce'],
                'val_dice': val_metrics['val_dice'],
                'val_iou': val_metrics['val_iou'],
                'val_bce_dice': val_metrics['val_bce_dice'],
                'grad_norm_epoch': epoch_grad_norm,
                'train_time_min': train_epoch_time,
                'val_time_min': val_epoch_time,
                'total_epoch_time_min': total_epoch_time,
            })

            try:
                log_validation_samples(model, val_loader, device, args.amp, experiment, epoch)
            except Exception as e:
                logging.error(f"Error logging validation samples: {e}")

            if args.save_checkpoint:
                ckpt_path = checkpoints_dir / f'checkpoint_epoch{epoch}.pth'
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f'Checkpoint {epoch} saved to {ckpt_path}')

            try:
                log_train_augment_preview(train_loader.dataset, fixed_indices_to_track, epoch, experiment)
            except Exception as e:
                logging.error(f"Error logging training augmentations: {e}")

        total_training_time = (time.time() - total_training_start) / 60.0
        experiment.log({'total_training_time_min': total_training_time})
        logging.info(f"Total training time: {total_training_time:.2f} minutes")




def main():
    """
    Main function to orchestrate the training process.
    """
    args = get_args()
    setup_logging(output_dir=args.output_dir)
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