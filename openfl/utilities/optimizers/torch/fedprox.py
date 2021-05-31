"""PyTorch FedProx optimizer module."""

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


class FedProxOptimizer(Optimizer):
    """FedProx optimizer.

    Paper: https://arxiv.org/pdf/1812.06127.pdf
    """

    def __init__(self,
                 params,
                 lr=required,
                 mu=0.0,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        """Initialize."""
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if mu < 0.0:
            raise ValueError("Invalid mu value: {}".format(mu))
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            momentum=momentum,
            nesterov=nesterov,
            dampening=dampening)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(FedProxOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super(FedProxOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            mu = group['mu']
            w_old = group['w_old']
            for p, w_old_p in zip(group['params'], w_old):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if w_old is not None:
                    d_p.add_(p - w_old_p, alpha=mu)
                p.add_(d_p, alpha=-group['lr'])

        return loss

    def set_old_weights(self, old_weights):
        """Set the global weights parameter to `old_weights` value."""
        for param_group in self.param_groups:
            param_group['w_old'] = old_weights
