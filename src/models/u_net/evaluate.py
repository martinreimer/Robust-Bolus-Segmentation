import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_loss = 0.0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred_sigmoid = F.sigmoid(mask_pred)

            # Compute loss
            loss = criterion(mask_pred, mask_true) + dice_coeff(mask_pred_sigmoid, mask_true)

            # Compute the Dice score
            dice_score += dice_coeff((mask_pred_sigmoid > 0.5).float(), mask_true, reduce_batch_first=False)

            total_loss += loss.item()

    net.train()
    return dice_score / max(num_val_batches, 1), total_loss / max(num_val_batches, 1)
