"""Implementation of FedCurv algorithm."""

from copy import deepcopy

import torch
import torch.nn.functional as F


def register_buffer(module: torch.nn.Module, name: str, value: torch.Tensor):
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
        raise AttributeError(f'{mod._get_name()} has no attribute `{buffer_name}`')

    buffer: torch.Tensor = getattr(mod, buffer_name)

    if buffer_name not in mod._buffers:
        raise AttributeError('`' + buffer_name + '` is not a buffer')

    return buffer


class FedCurv:
    """Federated Curvature class.

    Requires torch>=1.9.0.
    """

    def __init__(self, model: torch.nn.Module, importance: float):
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

    def _diag_fisher(self, model, data_loader, device):
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
                    precision_matrices[n].data += p.grad.data ** 2 / len(data_loader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def get_penalty(self, model):
        """Calculate the penalty term for the loss function.

        Args:
            model(torch.nn.Module): Model that stores global u_t and v_t values as buffers.

        Returns:
            float: Penalty term.
        """
        loss = 0
        if not self._params:
            return loss
        for n, p in model.named_parameters():
            if p.requires_grad:
                u_global, v_global = (get_buffer(model, target) for target in (f'{n}_u', f'{n}_v'))
                u_local, v_local = (getattr(self, name) for name in (f'{n}_u', f'{n}_v'))
                u = u_global - u_local
                v = v_global - v_local
                _loss = p ** 2 * u - 2 * p * v
                loss += _loss.sum()
        print(f'FedCurv penalty = {loss}')
        return self.importance * loss

    def on_train_begin(self, model):
        """Pre-train steps.

        Args:
            model(torch.nn.Module): model for training.
        """
        self._update_params(model)

    def on_train_end(self, model, data_loader, device):
        """Post-train steps.

        Args:
            model(torch.nn.Module): Trained model.
            data_loader(Iterable): Train dataset iterator.
            device(str): Model device.
            loss_fn(Callable): Train loss function.
        """
        precision_matrices = self._diag_fisher(model, data_loader, device)
        for n, m in precision_matrices.items():
            u = torch.tensor(m).to(device)
            v = torch.tensor(m) * model.get_parameter(n)
            v = v.to(device)
            register_buffer(model, f'{n}_u', u)
            register_buffer(model, f'{n}_v', v)
            setattr(self, f'{n}_u', u)
            setattr(self, f'{n}_v', v)
