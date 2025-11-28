# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Implementation of FedCurv algorithm."""

from copy import deepcopy

import torch
import torch.nn.functional as F


def register_buffer(module: torch.nn.Module, name: str, value: torch.Tensor):
    """Add a buffer to a module.

    Args:
        module (torch.nn.Module): The module to add the buffer to.
        name (str): The name of the buffer. Supports complex module names like
            'model.conv1.bias'.
        value (torch.Tensor): The value of the buffer.
    """
    module_path, _, name = name.rpartition(".")
    mod = module.get_submodule(module_path)
    mod.register_buffer(name, value)


def get_buffer(module, target):
    """Get a buffer from a module.

    Remove after pinning to a version
    where https://github.com/pytorch/pytorch/pull/61429 is included.
    Use module.get_buffer() instead.

    Args:
        module (torch.nn.Module): The module to get the buffer from.
        target (str): The name of the buffer to get.

    Returns:
        torch.Tensor: The buffer.
    """
    module_path, _, buffer_name = target.rpartition(".")

    mod: torch.nn.Module = module.get_submodule(module_path)

    if not hasattr(mod, buffer_name):
        raise AttributeError(f"{mod._get_name()} has no attribute `{buffer_name}`")

    buffer: torch.Tensor = getattr(mod, buffer_name)

    if buffer_name not in mod._buffers:
        raise AttributeError("`" + buffer_name + "` is not a buffer")

    return buffer


class FedCurv:
    """Federated Curvature class.

    This class implements the FedCurv algorithm for federated learning.
    Requires torch>=1.9.0.

    Args:
        model (torch.nn.Module): The base model. Parameters of it are used in
            loss penalty calculation.
        importance (float): The lambda coefficient of the FedCurv algorithm.
    """

    def __init__(self, model: torch.nn.Module, importance: float):
        """Initialize the FedCurv object.

        Args:
            model (torch.nn.Module): The base model. Parameters of it are used
                in loss penalty calculation.
            importance (float): The lambda coefficient of the FedCurv
                algorithm.
        """
        self.importance = importance
        self._params = {}
        self._register_fisher_parameters(model)

    def _register_fisher_parameters(self, model):
        """Register the Fisher parameters of the model.

        Args:
            model (torch.nn.Module): The model to register the Fisher
                parameters for.
        """
        params = list(model.named_parameters())
        for n, p in params:
            u = torch.zeros_like(p, requires_grad=False)
            v = torch.zeros_like(p, requires_grad=False)
            w = torch.zeros_like(p, requires_grad=False)

            # Add buffers to model for aggregation
            register_buffer(model, f"{n}_u", u)
            register_buffer(model, f"{n}_v", v)
            register_buffer(model, f"{n}_w", w)

            # Store buffers locally for subtraction in loss function
            setattr(self, f"{n}_u", u)
            setattr(self, f"{n}_v", v)
            setattr(self, f"{n}_w", w)

    def _update_params(self, model):
        """Update the parameters of the model.

        Args:
            model (torch.nn.Module): The model to update the parameters for.
        """
        self._params = deepcopy({n: p for n, p in model.named_parameters() if p.requires_grad})

    def _diag_fisher(self, model, data_loader, device):
        """Calculate the diagonal of the Fisher information matrix.

        Args:
            model (torch.nn.Module): The model to calculate the Fisher
                information matrix for.
            data_loader (Iterable): The data loader for the training data.
            device (str): The device to perform the calculations on.

        Returns:
            dict: The diagonal of the Fisher information matrix.
        """
        precision_matrices = {}
        for n, p in self._params.items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(device)

        model.eval()
        model.to(device)
        for sample, target in data_loader:
            model.zero_grad()
            sample = sample.to(device)
            target = target.to(device)
            output = model(sample)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data = p.grad.data**2 / len(data_loader)

        return precision_matrices

    def get_penalty(self, model):
        """Calculate the penalty term for the loss function.

        Args:
            model (torch.nn.Module): The model to calculate the penalty for.
                Stores global u_t and v_t values as buffers.

        Returns:
            float: The penalty term.
        """
        penalty = 0
        if not self._params:
            return penalty
        for name, param in model.named_parameters():
            if param.requires_grad:
                u_global, v_global, w_global = (
                    get_buffer(model, target).detach()
                    for target in (f"{name}_u", f"{name}_v", f"{name}_w")
                )
                u_local, v_local, w_local = (
                    getattr(self, name).detach() for name in (f"{name}_u", f"{name}_v", f"{name}_w")
                )
                u = u_global - u_local
                v = v_global - v_local
                w = w_global - w_local
                _penalty = param**2 * u - 2 * param * v + w
                penalty += _penalty.sum()
        penalty = self.importance * penalty
        return penalty.float()

    def on_train_begin(self, model):
        """Pre-train steps.

        Args:
            model (torch.nn.Module): The model for training.
        """
        self._update_params(model)

    def on_train_end(self, model: torch.nn.Module, data_loader, device):
        """Perform post-training steps.

        Args:
            model (torch.nn.Module): The trained model.
            data_loader (Iterable): The data loader for the training data.
            device (str): The device that the model was trained on.
        """
        precision_matrices = self._diag_fisher(model, data_loader, device)
        for n, m in precision_matrices.items():
            u = m.data.to(device)
            v = m.data * model.get_parameter(n)
            v = v.to(device)
            w = m.data * model.get_parameter(n) ** 2
            w = w.to(device)
            register_buffer(model, f"{n}_u", u.clone().detach())
            register_buffer(model, f"{n}_v", v.clone().detach())
            register_buffer(model, f"{n}_w", w.clone().detach())
            setattr(self, f"{n}_u", u.clone().detach())
            setattr(self, f"{n}_v", v.clone().detach())
            setattr(self, f"{n}_w", w.clone().detach())
