# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch FedProx optimizer module."""

import math

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
            raise ValueError(f'Invalid momentum value: {momentum}')
        if lr is not required and lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if mu < 0.0:
            raise ValueError(f'Invalid mu value: {mu}')
        defaults = {
            'dampening': dampening,
            'lr': lr,
            'momentum': momentum,
            'mu': mu,
            'nesterov': nesterov,
            'weight_decay': weight_decay,
        }

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov momentum requires a momentum and zero dampening')

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


class FedProxAdam(Optimizer):
    """FedProxAdam optimizer."""

    def __init__(self, params, mu=0, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        """Initialize."""
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if mu < 0.0:
            raise ValueError(f'Invalid mu value: {mu}')
        defaults = {'lr': lr, 'betas': betas, 'eps': eps,
                    'weight_decay': weight_decay, 'amsgrad': amsgrad, 'mu': mu}
        super(FedProxAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super(FedProxAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def set_old_weights(self, old_weights):
        """Set the global weights parameter to `old_weights` value."""
        for param_group in self.param_groups:
            param_group['w_old'] = old_weights

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, '
                            'please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adam(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      group['amsgrad'],
                      beta1,
                      beta2,
                      group['lr'],
                      group['weight_decay'],
                      group['eps'],
                      group['mu'],
                      group['w_old']
                      )
        return loss

    def adam(self, params,
             grads,
             exp_avgs,
             exp_avg_sqs,
             max_exp_avg_sqs,
             state_steps,
             amsgrad,
             beta1: float,
             beta2: float,
             lr: float,
             weight_decay: float,
             eps: float,
             mu: float,
             w_old):
        """Updtae optimizer parameters."""
        for i, param in enumerate(params):
            w_old_p = w_old[i]
            grad = grads[i]
            grad.add_(param - w_old_p, alpha=mu)
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)
