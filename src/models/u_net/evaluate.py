import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import time
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.dice_score import dice_loss

def compute_iou(mask_pred, mask_true, threshold=0.5, smooth=1e-6):
    """
    Compute Intersection-over-Union (IoU) for a batch.
    Assumes mask_pred has shape (N, 1, H, W).
    """
    pred_bin = (torch.sigmoid(mask_pred) > threshold).float()
    # Sum over channel, height, and width.
    intersection = (pred_bin * mask_true).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + mask_true.sum(dim=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()



@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion, pos_weight=None):
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0.0
    total_weighted_ce = 0.0
    total_dice = 0.0
    total_iou = 0.0

    # Create weighted cross entropy function if pos_weight is provided.
    weighted_ce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device)) if pos_weight is not None else nn.BCEWithLogitsLoss()

    t0 = time.time()
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            mask_pred = net(image)

            loss = criterion(mask_pred, mask_true)
            total_loss += loss.item()

            weighted_ce = weighted_ce_fn(mask_pred, mask_true)
            total_weighted_ce += weighted_ce.item()

            dice_metric = 1 - dice_loss(torch.sigmoid(mask_pred), mask_true, multiclass=False)
            total_dice += dice_metric.item()

            iou_metric = compute_iou(mask_pred, mask_true, threshold=0.5)
            total_iou += iou_metric

    val_time = (time.time() - t0) / 60.0

    net.train()
    avg_loss = total_loss / max(num_val_batches, 1)
    avg_weighted_ce = total_weighted_ce / max(num_val_batches, 1)
    avg_dice = total_dice / max(num_val_batches, 1)
    avg_iou = total_iou / max(num_val_batches, 1)
    avg_dice_bce = avg_dice + avg_weighted_ce

    return {
        'val_loss': avg_loss,
        'val_weighted_ce': avg_weighted_ce,
        'val_dice': avg_dice,
        'val_iou': avg_iou,
        'val_time': val_time,
        'val_bce_dice': avg_dice_bce
    }