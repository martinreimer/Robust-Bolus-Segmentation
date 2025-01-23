import torch
from torch import Tensor


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
