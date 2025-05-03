import torch
from torch import Tensor
import torch.nn as nn

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Computes the Dice coefficient for single-channel predictions vs. targets.
    By default, if reduce_batch_first=True, we expect (N,H,W) input.
    If input is (N,1,H,W), it automatically squeezes out the channel dimension.

    :param input:  model output or predicted mask
    :param target: ground truth mask
    :param reduce_batch_first: if True, sums over H,W for each item in batch
    :param epsilon: small constant to avoid division by zero
    """
    # --- Handle single-channel 4D (N,1,H,W) by squeezing channel dim ---
    if input.dim() == 4 and input.shape[1] == 1:
        input = input.squeeze(dim=1)  # => (N,H,W)
        target = target.squeeze(dim=1)  # => (N,H,W)


    # Now we expect them to match
    assert input.size() == target.size(), "Input and target must have the same shape!"

    # The original assertion: if reduce_batch_first=True, then input should be 3D
    # (batch, height, width). If input is 2D, the code handles a single mask (H, W).
    assert input.dim() == 3 or not reduce_batch_first, (
        f"Expected 3D if reduce_batch_first=True, got input.dim() = {input.dim()}"
    )

    # Determine which dimensions to sum over
    # If reduce_batch_first=True and input is 3D => sum over (H,W).
    # If input is 2D or reduce_batch_first=False => sum over all dims except possibly batch.
    sum_dim = (-1, -2) if (input.dim() == 2 or not reduce_batch_first) else (-1, -2, -3)

    inter = 2.0 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    # Avoid dividing by zero if sets_sum is zero
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor,
                          reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Multi-class version: flattens out the class dimension so dice_coeff
    can be computed over each channel, then averaged.
    """
    # Flatten the (N,C,H,W) => (N*C,H,W) so that dice_coeff can be reused
    return dice_coeff(
        input.flatten(0, 1),
        target.flatten(0, 1),
        reduce_batch_first=reduce_batch_first,
        epsilon=epsilon
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """
    Dice loss = 1 - Dice coefficient.
    By default, assume binary segmentation (multiclass=False).
    For multi-class, uses the multiclass_dice_coeff above.
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    # We set reduce_batch_first=True by default so that for each sample in a batch
    # we compute dice, then average over the batch.
    return 1 - fn(input, target, reduce_batch_first=True)

from typing import Optional

class DiceLoss(nn.Module):
    """
    Dice loss implemented as a `torch.nn.Module`.

    Parameters
    ----------
    multiclass : bool, optional (default=False)
        If *False* ⇒ binary Dice on a single channel.
        If *True*  ⇒ multiclass Dice (averaged over C channels).
    from_logits : bool, optional (default=True)
        If *True* the forward expects **unnormalised** logits and applies
        `sigmoid` internally; set to *False* if you already pass probabilities.
    reduce_batch_first : bool, optional (default=True)
        Average Dice over each item in the batch before taking the mean.
    epsilon : float, optional
        Smoothing term to avoid division by zero.
    """
    def __init__(self,
                 multiclass: bool = False,
                 from_logits: bool = True,
                 reduce_batch_first: bool = True,
                 epsilon: float = 1e-6):
        super().__init__()
        self.multiclass = multiclass
        self.from_logits = from_logits
        self.reduce_batch_first = reduce_batch_first
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Model output of shape (N, C, H, W) or (N, 1, H, W).
        target : torch.Tensor
            Ground-truth mask with the same shape.

        Returns
        -------
        torch.Tensor
            Scalar Dice loss.
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)

        fn = multiclass_dice_coeff if self.multiclass else dice_coeff
        dice = fn(pred, target,
                  reduce_batch_first=self.reduce_batch_first,
                  epsilon=self.epsilon)
        return 1.0 - dice


import torch
import torch.nn as nn

class IoULoss(nn.Module):
    r"""
    Jaccard / IoU loss implemented as a `torch.nn.Module`.

    L = 1 – IoU
    IoU = TP / (TP + FP + FN)

    Parameters
    ----------
    multiclass : bool, optional (default=False)
        `False` → binary IoU on one channel.
        `True`  → compute IoU per class and return the **mean**.
    from_logits : bool (default=True)
        If `True` the forward expects **unnormalised** logits and will
        `sigmoid` them internally; set `False` if you already supply
        probabilities.
    smooth : float, optional
        Smoothing constant added to numerator & denominator.
    reduce_batch_first : bool, optional (default=True)
        Average over items in the batch before taking the final mean.
    """
    def __init__(self,
                 multiclass: bool = False,
                 from_logits: bool = True,
                 smooth: float = 1e-6,
                 reduce_batch_first: bool = True):
        super().__init__()
        self.multiclass = multiclass
        self.from_logits = from_logits
        self.smooth = smooth
        self.reduce_batch_first = reduce_batch_first

    def _binary_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """IoU for a single (probabilistic) mask pair."""
        inter = (pred * target).sum(dim=(-1, -2))
        union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - inter
        return (inter + self.smooth) / (union + self.smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (N, C, H, W) or (N, 1, H, W)
        target : same shape as `pred`
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)

        if self.multiclass:
            # iterate over channels and average
            num_classes = pred.size(1)
            ious = []
            for c in range(num_classes):
                ious.append(self._binary_iou(pred[:, c], target[:, c]))
            iou = torch.stack(ious, dim=0).mean()
        else:  # binary
            if pred.size(1) == 1:                # squeeze channel dim
                pred, target = pred[:, 0], target[:, 0]
            iou = self._binary_iou(pred, target)
            if self.reduce_batch_first:
                iou = iou.mean()

        return 1.0 - iou



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # controls class imbalance
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate Binary Cross-Entropy Loss for each sample
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute pt (model confidence on true class)
        pt = torch.exp(-BCE_loss)

        # Apply the focal adjustment
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, dice_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # Focal Loss
        focal_loss = self.focal_loss(inputs, targets)

        # Dice Loss
        smooth = 1e-6  # Smooth to prevent division by zero
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Weighted Sum of Focal and Dice Loss
        return focal_loss + self.dice_weight * dice_loss


class LogCoshDiceLoss(nn.Module):
    """
    Log-cosh Dice loss  L = log(cosh(DiceLoss))
    Works for binary or multiclass depending on the dice_loss you pass.
    """
    def __init__(self, multiclass: bool = False):
        super().__init__()
        self.multiclass = multiclass

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        d = dice_loss(torch.sigmoid(pred), target, multiclass=self.multiclass)
        # torch.cosh operates element-wise
        return torch.log(torch.cosh(d))


import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    r"""
    Tversky loss for binary segmentation

    L = 1 - TP / (TP + α·FN + β·FP)

    Args
    ----
    alpha (float) – weight for false-negatives
    beta  (float) – weight for false-positives
    smooth (float) – small constant to avoid division by zero
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.sigmoid(logits)
        probs_flat  = probs.view(-1)
        target_flat = target.view(-1)

        tp = (probs_flat * target_flat).sum()
        fn = ((1 - probs_flat) * target_flat).sum()
        fp = (probs_flat * (1 - target_flat)).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        return 1 - tversky