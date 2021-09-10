from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from openfl.utilities import Metric


def add_param(module, name, value):
    if '.' in name:
        submodule, name = name.split('.', 1)
        add_param(getattr(module, submodule), name, value)
    else:
        module.register_parameter(name, nn.Parameter(value, requires_grad=False))

class FedCurv(nn.Module):
    def __init__(self, importance):
        super().__init__()
        self.importance = importance
        self._params = {}

    def register_fisher_parameters(self):  # Should be called last in child constructor
        for n, p in deepcopy(self).named_parameters():
            add_param(self, f'{n}_fisher', torch.zeros_like(p))
    
    def update_params(self):
        self._params = deepcopy({n: p for n, p in self.named_parameters() if p.requires_grad})

    @property
    def params(self):
        return self._params
    
    def _diag_fisher(self, samples, targets, device):
        precision_matrices = {}
        for n, p in self.params.items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(device)

        self.eval()
        for sample, target in zip(samples, targets):
            self.zero_grad()
            sample = sample.to(device)
            output = self(sample).view(1, -1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()

            for n, p in self.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2 / len(samples)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self):
        loss = 0
        if self.params is None:
            return loss
        for n, p in self.named_parameters():
            if p.requires_grad:
                _loss = self.get_parameter(f'{n}_fisher') * (p - self.params.get(n, p)) ** 2
                loss += _loss.sum()
        return self.importance * loss

    def train_epoch(self, batch_generator):
        losses = []
        samples = []
        targets = []
        for data, target in batch_generator:
            data, target = torch.tensor(data).to(self.device), torch.tensor(target).to(self.device)
            samples += [x.unsqueeze(0) for x in data]
            targets += [t.argmax().unsqueeze(0) for t in target]
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output=output, target=target) + self.model.penalty()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        self.model.update_params()
        precision_matrices = self.model._diag_fisher(samples, targets, self.device)
        for n, m in precision_matrices.items():
            add_param(self.model, f'{n}_fisher', torch.tensor(m))
        return Metric(self.loss_fn.__name__, np.array(loss))
