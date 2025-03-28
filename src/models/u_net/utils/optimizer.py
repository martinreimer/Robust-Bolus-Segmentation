import torch
import argparse

def get_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimizer_choice = args.optimizer.lower()
    if optimizer_choice == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay
        )
    elif optimizer_choice == 'nadam':
        return torch.optim.NAdam(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay
        )
    elif optimizer_choice == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.rmsprop_weight_decay,
            momentum=args.rmsprop_momentum,
            foreach=True
        )
    elif optimizer_choice == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.sgd_momentum,
            nesterov=args.sgd_nesterov
        )
    elif optimizer_choice == 'adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'adamax':
        return torch.optim.Adamax(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")
