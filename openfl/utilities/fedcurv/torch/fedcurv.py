"""Implementation of FedCurv algorithm."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn


def register_buffer(module: nn.Module, name: str, value: torch.Tensor):
    """Add a buffer to module.

    Args:
        module: Module
        name: Buffer name. Supports complex module names like 'model.conv1.bias'.
        value: Buffer value
    """
    module_path, _, name = name.rpartition('.')
    mod = module.get_submodule(module_path)
    mod.register_buffer(name, value)


def get_buffer(module, target):
    """Get module buffer.

    Remove after pinning to a version
    where https://github.com/pytorch/pytorch/pull/61429 is included.
    Use module.get_buffer() instead.
    """
    module_path, _, buffer_name = target.rpartition('.')

    mod: torch.nn.Module = module.get_submodule(module_path)

    if not hasattr(mod, buffer_name):
        raise AttributeError(mod._get_name() + ' has no attribute `'
                                + buffer_name + '`')

    buffer: torch.Tensor = getattr(mod, buffer_name)

    if buffer_name not in mod._buffers:
        raise AttributeError('`' + buffer_name + '` is not a buffer')

    return buffer


class FedCurv:
    """Federated Curvature class.

    Create instance of it locally and replace your training loop with
    train() function of the instance.

    IMPORTANT: In order to use pure FedCurv you also need to override aggregation function
    with openfl.utilities.fedcurv.aggregation_function.FedCurvWeightedAverage.

    Requires torch>=1.9.0.
    """

    def __init__(self, model: nn.Module, importance: float):
        """Initialize.
        
        Args:
            model: Base model. Parameters of it are used in loss penalty calculation.
            importance: Lambda coefficient of FedCurv algorithm.
            """
        self.importance = importance
        self._params = {}
        self._register_fisher_parameters(model)

    def _register_fisher_parameters(self, model):
        params = list(model.named_parameters())
        for n, p in params:
            u = torch.zeros_like(p, requires_grad=False)
            v = torch.zeros_like(p, requires_grad=False)

            # Add buffers to model for aggregation
            register_buffer(model, f'{n}_u', u)
            register_buffer(model, f'{n}_v', v)

            # Store buffers locally for subtraction in loss function
            setattr(self, f'{n}_u', u)
            setattr(self, f'{n}_v', v)

    def _update_params(self, model):
        self._params = deepcopy({n: p for n, p in model.named_parameters() if p.requires_grad})

    def _diag_fisher(self, model, samples, targets, device):
        precision_matrices = {}
        for n, p in self._params.items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(device)

        model.eval()
        model.to(device)
        for sample, target in zip(samples, targets):
            model.zero_grad()
            sample = sample.to(device)
            output = model(sample).view(1, -1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2 / len(samples)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def _penalty(self, model):
        loss = 0
        if not self._params:
            return loss
        for n, p in model.named_parameters():
            if p.requires_grad:
                u_global, v_global = (get_buffer(model, target).detach() for target in (f'{n}_u', f'{n}_v'))
                u_local, v_local = (getattr(self, name).detach() for name in (f'{n}_u', f'{n}_v'))
                u = u_global - u_local
                v = v_global - v_local
                _loss = p ** 2 * u - 2 * p * v
                loss += _loss.sum()
        return self.importance * loss

    def train(self, model, data_loader, optimizer, device, loss_fn):
        """Training loop.
        
        This function has the same signature as train functions registered in TaskInterface.
        """
        device = 'cpu'  # force CPU training for full reproducibility
        data_loader = tqdm.tqdm(data_loader, desc='train')

        model.train()
        model.to(device)
        losses = []
        samples = []
        targets = []
        for data, target in data_loader:
            data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
            samples += [x.unsqueeze(0) for x in data]
            targets += [t.argmax().unsqueeze(0) for t in target]
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output=output, target=target) + self._penalty(model)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        self._update_params(model)
        precision_matrices = self._diag_fisher(model, samples, targets, device)
        for n, m in precision_matrices.items():
            u = torch.tensor(m)
            v = torch.tensor(m) * model.get_parameter(n)
            register_buffer(model, f'{n}_u', u)
            register_buffer(model, f'{n}_v', v)
            setattr(self, f'{n}_u', u)
            setattr(self, f'{n}_v', v)
        return {loss_fn.__name__: np.array(loss)}
