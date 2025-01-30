'''
 python train.py -c 1 -e 3 -b 30
'''

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
torch.backends.cudnn.benchmark = True # if input size is fixed, it boosts the performance
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
import albumentations as A
import numpy as np

dataset_path = '../../../data/processed/dataset_first_experiments'#dataset_train_val_test_split/'
#dataset_path = '../../../data/foreback/processed'#dataset_train_val_test_split/'

python train.py -e 1 -b 30 --train-samples 10 --val-samples 10 --dataset-path ../../../data/processed/dataset_first_experiments


def log_train_augment_preview(dataset, fixed_indices, epoch, experiment, count=3):
    for idx in fixed_indices:
        sample = dataset[idx]
        img_tensor = sample['image'].to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, 1, H, W]
        mask_tensor = sample['mask'].to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, 1, H, W]

        # Convert tensors to CPU numpy arrays
        img_np = img_tensor.squeeze().cpu().numpy()  # [H, W]
        mask_np = mask_tensor.squeeze().cpu().numpy()  # [H, W]

        # Log to WandB with consistent naming
        wandb.log({
            f"train_augments/sample_{idx}/epoch_{epoch}": [
                wandb.Image(img_np, caption=f"Sample {idx} - Augmented Input"),
                wandb.Image(mask_np, caption=f"Sample {idx} - Augmented Mask"),
            ]
        })


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,

):
    # === Create a run directory ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("./runs") / f"run-{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Checkpoints and final model will be saved to: {checkpoints_dir}")

    # 1. Create datasets for predefined splits
    base_dir = dataset_path

    # add param for subsset and transform

    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5, value=0, mask_value=0),
        A.RandomBrightnessContrast(p=0.1, brightness_limit=0.025, contrast_limit=0.025),
        #A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        #A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        #A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3),

        #A.GridDistortion(p=0.3),
        #A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        #A.Affine(scale={"min": 0.9, "max": 1.1}, rotate={"min": -10, "max": 10},
        #         translate_percent={"x": 0.05, "y": 0.05}, p=0.5),
    ], additional_targets={'mask': 'mask'})

    train_set = BasicDataset(
        base_dir=dataset_path,
        subset='train',
        mask_suffix='_bolus',
        transform=augmentations
    )
    val_set = BasicDataset(
        base_dir=dataset_path,
        subset='val',
        mask_suffix='_bolus',
        transform=None
    )

    # Select only 20 images for training and validation
    #train_set = torch.utils.data.Subset(train_set, range(min(5, len(train_set))))
    #val_set = torch.utils.data.Subset(val_set, range(min(5, len(val_set))))

    n_train = len(train_set)
    n_val = len(val_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=6, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging / wandb)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, amp=amp)
    )

    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    """)

    # 4. Set up the optimizer, the loss, and the scheduler
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # maximize Dice
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)

    # For binary segmentation: BCE + Dice
    criterion = nn.BCEWithLogitsLoss()

    global_step = 0

    # to track the augmentations of the same images: define indices to track
    fixed_indices_to_track = [0, len(train_set)//2, len(train_set)-1]

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            sample_count_train = 0
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']


                print(f"sample count: {sample_count_train}")
                sample_count_train += 1
                # images: [N, 1, H, W]
                # true_masks: [N, 1, H, W] (0/1)
                assert images.shape[1] == model.n_channels, \
                    f'Network defined with {model.n_channels} channels, got {images.shape[1]}.'

                # Move to device
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss_bce = criterion(masks_pred, true_masks)
                    loss_dice = dice_loss(F.sigmoid(masks_pred), true_masks, multiclass=False)
                    loss = loss_bce + loss_dice

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                # Unscale grads and compute grad norm
                grad_scaler.unscale_(optimizer)

                total_norm = 0.0

                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5  # L2 norm of the gradients
                epoch_grad_norm += total_norm


                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Log training loss *per batch*

                experiment.log({
                    'train_loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

        # --- End of epoch: compute metrics on the entire epoch ---
        # 1) Average training loss
        epoch_loss /= max(num_batches, 1)
        # 2) Average gradient norm
        epoch_grad_norm /= max(num_batches, 1)

        # --- Validation (Dice & Loss) Once Per Epoch ---
        val_dice, val_loss = evaluate(
            net=model,
            dataloader=val_loader,
            device=device,
            amp=amp,
            criterion=criterion
        )
        scheduler.step(val_dice)  # reduce LR on plateau, using Dice as "metric"

        logging.info(f"""Epoch {epoch} / {epochs}:
            Train Loss: {epoch_loss:.4f}
            Val Dice:   {val_dice:.4f}
            Val Loss:   {val_loss:.4f}
            Grad Norm:  {epoch_grad_norm:.4f}
        """)

        # --- Log epoch-level metrics to WandB ---
        experiment.log({
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'train_loss_epoch': epoch_loss,
            'val_dice_epoch': val_dice,
            'val_loss_epoch': val_loss,
            'grad_norm_epoch': epoch_grad_norm
        })

        # --- Log sample images from validation for debugging/visualization ---
        # Evaluate again, but only store first few predictions
        # We'll reuse the images from the evaluate function or do a short pass ourselves

        # Let's do a small pass over val_loader to get up to 3 samples
        model.eval()
        sample_count = 0
        for batch in val_loader:
            img, true_mask = batch['image'], batch['mask']
            img = img.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=amp):
                pred_mask = model(img)
            pred_mask_bin = (torch.sigmoid(pred_mask) > 0.5).float()

            for i in range(img.shape[0]):
                # Only log up to 3 examples total
                if sample_count >= 3:
                    break

                # Prepare image for logging
                wandb.log({
                    f"examples/epoch_{epoch}_sample_{sample_count}": [
                        wandb.Image(img[i].cpu(), caption="Input"),
                        wandb.Image(true_mask[i].cpu(), caption="True Mask"),
                        wandb.Image(pred_mask_bin[i].cpu(), caption="Predicted Mask"),
                    ]
                })
                sample_count += 1

            if sample_count >= 3:
                break

        # --- Save checkpoint ---
        if save_checkpoint:
            ckpt_path = checkpoints_dir / f'checkpoint_epoch{epoch}.pth'
            state_dict = model.state_dict()
            torch.save(state_dict, str(ckpt_path))
            logging.info(f'Checkpoint {epoch} saved to {ckpt_path}')

        # --- Log training augmentations ---
        log_train_augment_preview(train_set, fixed_indices_to_track, epoch, experiment, count=3)


    # === After training, save final model ===
    final_model_path = checkpoints_dir / 'model.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Final model saved to {final_model_path}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision') #always true
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    print(f'Using device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'Current device: {torch.cuda.current_device()}')

    # Build model
    model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last).to(device=device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Load weights if requested
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    # Train
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! Enabling checkpointing for gradient to reduce memory usage.')
        torch.cuda.empty_cache()
        model.use_checkpointing()  # If your model supports gradient checkpointing
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            amp=args.amp
        )