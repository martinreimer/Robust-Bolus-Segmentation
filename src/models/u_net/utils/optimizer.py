import torch
import argparse


def get_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimizer_choice = args.optimizer.lower()
    if optimizer_choice == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'nadam':
        # PyTorch 1.13+ has torch.optim.NAdam. If you are on an earlier version, consider upgrading or using Adam.
        return torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(),
                                   lr=args.lr,
                                   weight_decay=1e-8,
                                   momentum=0.999,
                                   foreach=True)
    elif optimizer_choice == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif optimizer_choice == 'adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'adamax':
        return torch.optim.Adamax(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")
