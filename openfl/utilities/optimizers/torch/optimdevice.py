"""PyTorch optimizer utility to transfer the optimizer's parametes to current device."""

import torch


def optimizer_to(optim, device):
    """Transfer optimizer to device.

    Transfer optimizer's parameters to the same device where the model is.
    This is because optimized parameters must live in consistent locations
    when optimizers are constructed and used.
    Resembles model.to(device) for optimizer.

    Usage (in train task after the model is transferred to device):
            optimizer_to(optimizer,device)
    Args:
        optimizer   :  Torch optimizer
        device      :  model's device
    """
    for param in optim.state.values():
        for subparam in param.values():
            if isinstance(subparam, torch.Tensor):
                subparam.data = subparam.data.to(device)
                if subparam._grad is not None:
                    subparam._grad.data = subparam._grad.data.to(device)
