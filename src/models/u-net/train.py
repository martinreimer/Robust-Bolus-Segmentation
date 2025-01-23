import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datetime import datetime
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchviz import make_dot
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

# Remove the global "dir_checkpoint"
dataset_path = '../../../data/processed/dataset_first_experiments/train/'
dir_img = Path(dataset_path + 'imgs/')
dir_mask = Path(dataset_path + 'masks/')
# dir_checkpoint = Path('./checkpoints/')  # <--- REMOVED

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # === Create a run directory ===
    # E.g. "runs/run-20250123_101130"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("./runs") / f"run-{timestamp}"

    # We create a 'checkpoints' subfolder for each run
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Checkpoints and final model will be saved to: {checkpoints_dir}")

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='_bolus')

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader   = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging / wandb)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint,
             img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, and the scheduler
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # maximize Dice
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)

    # Choose loss function
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # images: [N, model.n_channels, H, W]
                # true_masks: [N, 1, H, W] (binary) or [N, H, W] (multi-class)
                assert images.shape[1] == model.n_channels, \
                    f'Network defined with {model.n_channels} input channels, but loaded {images.shape[1]}.'

                # Move to device
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # For single-class => float target; multi-class => long target
                if model.n_classes == 1:
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                else:
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device_type=(device.type if device.type != 'mps' else 'cpu'), enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        # Binary
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(F.sigmoid(masks_pred), true_masks, multiclass=False)
                    else:
                        # Multi-class
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Periodically evaluate
                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and (global_step % division_step == 0):
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if (value.grad is not None) and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)

                    logging.info(f'Validation Dice score: {val_score}')
                    try:
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu())
                                  if model.n_classes > 1
                                  else wandb.Image(F.sigmoid(masks_pred)[0].cpu())
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                    except Exception as e:
                        logging.warning(f"Logging to W&B failed: {e}")

        # End of epoch: optionally save checkpoint
        if save_checkpoint:
            state_dict = model.state_dict()
            # If needed, store mask_values or other data
            if hasattr(dataset, 'mask_values'):
                state_dict['mask_values'] = dataset.mask_values

            ckpt_path = checkpoints_dir / f'checkpoint_epoch{epoch}.pth'
            torch.save(state_dict, str(ckpt_path))
            logging.info(f'Checkpoint {epoch} saved to {ckpt_path}')

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
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

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
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
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
            img_scale=args.scale,
            val_percent=args.val / 100,
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
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
