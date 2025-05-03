import time
from contextlib import nullcontext
import torch
from torch import nn

@torch.inference_mode()
def evaluate(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
    criterion: nn.Module,
    pos_weight: float = None,
    threshold: float = 0.5,
    eps: float = 1e-8,
):
    """
       Evaluate the network on the validation set, timing each stage to identify bottlenecks.
       Returns metric dict including detailed timing breakdowns.
       """
    net.eval()
    dtype = torch.float32

    # AMP context: only when on CUDA
    amp_ctx = torch.cuda.amp.autocast if device.type == 'cuda' else nullcontext

    # Secondary BCE loss
    bce_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    ) if pos_weight is not None else nn.BCEWithLogitsLoss()

    # Accumulators for metrics (float) and timing
    total_loss = total_bce = total_dice = total_iou = 0.0
    total_tp = total_fp = total_tn = total_fn = 0.0
    times = {
        'data_load': 0.0,
        'forward': 0.0,
        'metrics': 0.0,
    }

    # Overall timer
    epoch_start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            # 1) Data load + transfer
            t0 = time.time()
            img = batch['image'].to(device, dtype=dtype, memory_format=torch.channels_last)
            mask = batch['mask'].to(device, dtype=dtype)
            times['data_load'] += time.time() - t0

            # 2) Forward + loss
            t1 = time.time()
            with amp_ctx(enabled=amp):
                logits = net(img)
                loss = criterion(logits, mask)
                bce = bce_fn(logits, mask)
            times['forward'] += time.time() - t1

            # 3) Prediction and metric computations
            t2 = time.time()
            prob = torch.sigmoid(logits)
            preds = (prob > threshold).to(dtype)

            inter = (preds * mask).sum(dim=[1, 2, 3])
            p_sum = preds.sum(dim=[1, 2, 3])
            m_sum = mask.sum(dim=[1, 2, 3])
            union = p_sum + m_sum

            batch_dice = (2 * inter / (p_sum + m_sum + eps)).mean()
            batch_iou = (inter / (union - inter + eps)).mean()
            tp = inter.mean()
            fp = ((preds * (1 - mask)).sum(dim=[1, 2, 3])).mean()
            fn = (((1 - preds) * mask).sum(dim=[1, 2, 3])).mean()
            tn = (((1 - preds) * (1 - mask)).sum(dim=[1, 2, 3])).mean()
            times['metrics'] += time.time() - t2

            # Accumulate metrics
            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += batch_dice.item()
            total_iou += batch_iou.item()
            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
            total_tn += tn.item()

    # Total epoch time
    total_time = (time.time() - epoch_start) / 60.0
    net.train()

    # Normalize by batches
    N = float(max(len(dataloader), 1))

    # Derived metrics
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    specificity = total_tn / (total_tn + total_fp + eps)

    results = {
        'val_loss': total_loss / N,
        'val_weighted_ce': total_bce / N,
        'val_dice': total_dice / N,
        'val_iou': total_iou / N,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
        'val_accuracy': accuracy,
        'val_specificity': specificity,
        'val_time': total_time,
        # Timing breakdowns (in seconds)
        'val_time_data_load': times['data_load'],
        'val_time_forward': times['forward'],
        'val_time_metrics': times['metrics'],
    }
    return results
