# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""PyTorch FedProx optimizer module."""

import math

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


class FedProxOptimizer(Optimizer):
    """FedProx optimizer.

    Implements the FedProx optimization algorithm using PyTorch.
    FedProx is a federated learning optimization algorithm designed to handle
    non-IID data.
    It introduces a proximal term to the federated averaging algorithm to
    reduce the impact of devices with outlying updates.

    Paper: https://arxiv.org/pdf/1812.06127.pdf

    Attributes:
        params: Parameters to be stored for optimization.
        lr: Learning rate.
        mu: Proximal term coefficient.
        momentum: Momentum factor.
        dampening: Dampening for momentum.
        weight_decay: Weight decay (L2 penalty).
        nesterov: Enables Nesterov momentum.
    """

    def __init__(
        self,
        params,
        lr=required,
        mu=0.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        """
        Initialize the FedProx optimizer.

        Args:
            params: Parameters to be stored for optimization.
            lr: Learning rate.
            mu: Proximal term coefficient. Defaults to 0.0.
            momentum: Momentum factor. Defaults to 0.
            dampening: Dampening for momentum. Defaults to 0.
            weight_decay: Weight decay (L2 penalty). Defaults to 0.
            nesterov: Enables Nesterov momentum. Defaults to False

        Raises:
            ValueError: If momentum is less than 0.
            ValueError: If learning rate is less than 0.
            ValueError: If weight decay is less than 0.
            ValueError: If mu is less than 0.
        """
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if mu < 0.0:
            raise ValueError(f"Invalid mu value: {mu}")
        defaults = {
            "dampening": dampening,
            "lr": lr,
            "momentum": momentum,
            "mu": mu,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
        }

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        """
        Set optimizer state.

        Args:
            state: State dictionary.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Loss value if closure is provided. None otherwise.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            mu = group["mu"]
            w_old = group["w_old"]
            for p, w_old_p in zip(group["params"], w_old):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if w_old is not None:
                    d_p.add_(p - w_old_p, alpha=mu)
                p.add_(d_p, alpha=-group["lr"])

        return loss

    def set_old_weights(self, old_weights):
        """Set the global weights parameter to `old_weights` value.

        Args:
            old_weights: The old weights to be set.
        """
        for param_group in self.param_groups:
            param_group["w_old"] = old_weights


class FedProxAdam(Optimizer):
    """FedProxAdam optimizer.

    Implements the FedProx optimization algorithm with Adam optimizer.

    Attributes:
        params: Parameters to be stored for optimization.
        mu: Proximal term coefficient.
        lr: Learning rate.
        betas: Coefficients used for computing running averages of gradient
            and its square.
        eps: Value for computational stability.
        weight_decay: Weight decay (L2 penalty).
        amsgrad: Whether to use the AMSGrad variant of this algorithm.
    """

    def __init__(
        self,
        params,
        mu=0,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        """
        Args:
            params: Parameters to be stored for optimization.
            mu: Proximal term coefficient. Defaults to 0.
            lr: Learning rate. Defaults to 1e-3.
            betas: Coefficients used for computing running averages of
                gradient and its square. Defaults to (0.9, 0.999).
            eps: Value for computational stability. Defaults to 1e-8.
            weight_decay: Weight decay (L2 penalty). Defaults to 0.
            amsgrad: Whether to use the AMSGrad variant of this algorithm.
                Defaults to False.

        Raises:
            ValueError: If learning rate is less than 0.
            ValueError: If betas[0] is not in [0, 1).
            ValueError: If betas[1] is not in [0, 1).
            ValueError: If weight decay is less than 0.
            ValueError: If mu is less than 0.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if mu < 0.0:
            raise ValueError(f"Invalid mu value: {mu}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "mu": mu,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def set_old_weights(self, old_weights):
        """Set the global weights parameter to `old_weights` value.

        Args:
            old_weights: The old weights to be set.
        """
        for param_group in self.param_groups:
            param_group["w_old"] = old_weights

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Loss value if closure is provided. None otherwise.
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

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, "
                            "please consider SparseAdam instead"
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            beta1, beta2 = group["betas"]
            self.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group["amsgrad"],
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["mu"],
                group["w_old"],
            )
        return loss

    def adam(
        self,
        params,
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
        w_old,
    ):
        """
        Update optimizer parameters.

        Args:
            params: Parameters to be stored for optimization.
            grads: Gradients.
            exp_avgs: Exponential moving average of gradient values.
            exp_avg_sqs: Exponential moving average of squared gradient values.
            max_exp_avg_sqs: Maintains max of all exp. moving avg. of sq. grad. values.
            state_steps: Steps for each param group update.
            amsgrad: Whether to use the AMSGrad variant of this algorithm.
            beta1 (float): Coefficient used for computing running averages of
                gradient.
            beta2 (float): Coefficient used for computing running averages of
                squared gradient.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty).
            eps (float): Value for computational stability.
            mu (float): Proximal term coefficient.
            w_old: The old weights.
        """
        for i, param in enumerate(params):
            w_old_p = w_old[i]
            grad = grads[i]
            grad.add_(param - w_old_p, alpha=mu)
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

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
