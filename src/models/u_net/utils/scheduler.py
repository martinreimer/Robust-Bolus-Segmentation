import torch.optim.lr_scheduler
import torch.optim as optim
import argparse

def get_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace):
    """
    Returns a learning rate scheduler based on the provided arguments.
    Supports:
      - StepLR: Decays the LR at fixed intervals.
      - ExponentialLR: Decays the LR exponentially each epoch.
      - ReduceLROnPlateau: Adjusts the LR based on validation loss.
    """
    if args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.scheduler == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.lr_mode, patience=args.lr_patience, factor=args.lr_gamma)
    else:
        raise ValueError(f"Unsupported scheduler type: {args.scheduler}")
