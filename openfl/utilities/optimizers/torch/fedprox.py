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

    IMPORTANT: This optimizer requires a reference to the original (global) model parameters
    to calculate the proximal term. These must be set explicitly using the set_old_weights()
    method before training begins. The old weights (w_old) must match the order and structure
    of the model's parameters. Typically, w_old should be set to the initial global model
    parameters received from the aggregator at the beginning of each round.

    If mu > 0 and w_old is not set, the optimizer will raise a ValueError.

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
            import warnings

            warnings.warn(
                f"Negative mu value ({mu}) will cause the proximal term to reward "
                f"deviations from global weights, which may be counterintuitive.",
                UserWarning,
                stacklevel=2,
            )
        defaults = {
            "dampening": dampening,
            "lr": lr,
            "momentum": momentum,
            "mu": mu,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "w_old": None,  # Initialize w_old as None
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

    def _validate_old_weights(self, mu, w_old):
        """Validate old weights for FedProx regularization.

        Args:
            mu: Proximal term coefficient
            w_old: Old weights reference

        Raises:
            ValueError: If mu > 0 and w_old is None
        """
        if mu > 0 and w_old is None:
            raise ValueError(
                "FedProx requires old weights to be set when mu > 0. "
                "Please call set_old_weights() before optimization step."
            )

    def _apply_momentum(self, p, d_p, momentum, dampening, nesterov):
        """Apply momentum to gradient.

        Args:
            p: Parameter
            d_p: Gradient
            momentum: Momentum factor
            dampening: Dampening factor
            nesterov: Whether to use Nesterov momentum

        Returns:
            Modified gradient
        """
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
        return d_p

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

            # Validate old weights for FedProx
            self._validate_old_weights(mu, w_old)

            # Apply proximal term when mu != 0
            apply_proximal = w_old is not None

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                d_p = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Apply momentum
                if momentum != 0:
                    d_p = self._apply_momentum(p, d_p, momentum, dampening, nesterov)

                # Apply proximal term
                if apply_proximal:
                    w_old_p = w_old[i]
                    d_p.add_(p - w_old_p, alpha=mu)

                # Apply gradient step
                p.add_(d_p, alpha=-group["lr"])

        return loss

    def set_old_weights(self, old_weights):
        """Set the global weights parameter to `old_weights` value.

        This method must be called before training begins to set the reference point for
        calculating the proximal term in FedProx. Typically, this should be set to the
        initial global model parameters received from the aggregator at the beginning
        of each federated learning round.

        If mu > 0 and this method is not called, the optimizer will raise a ValueError
        during the optimization step.

        Args:
            old_weights: List of parameter tensors representing the global model weights.
                         Must match the order and structure of the model's parameters
                         being optimized (typically obtained by calling
                         [p.clone().detach() for p in model.parameters()]).
        """
        for param_group in self.param_groups:
            param_group["w_old"] = old_weights


class FedProxAdam(Optimizer):
    """FedProxAdam optimizer.

    Implements the FedProx optimization algorithm with Adam optimizer.

    IMPORTANT: This optimizer requires a reference to the original (global) model parameters
    to calculate the proximal term. These must be set explicitly using the set_old_weights()
    method before training begins. The old weights (w_old) must match the order and structure
    of the model's parameters. Typically, w_old should be set to the initial global model
    parameters received from the aggregator at the beginning of each round.

    If mu > 0 and w_old is not set, the optimizer will raise a ValueError.

    Attributes:
        params: Parameters to be stored for optimization.
        mu: Proximal term coefficient.
        lr: Learning rate.
        betas: Coefficients used for computing running averages of gradient and its square.
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
            import warnings

            warnings.warn(
                f"Negative mu value ({mu}) will cause the proximal term to reward "
                f"deviations from global weights, which may be counterintuitive.",
                UserWarning,
                stacklevel=2,
            )
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "mu": mu,
            "w_old": None,  # Initialize w_old as None
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def set_old_weights(self, old_weights):
        """Set the global weights parameter to `old_weights` value.

        This method must be called before training begins to set the reference point for
        calculating the proximal term in FedProx. Typically, this should be set to the
        initial global model parameters received from the aggregator at the beginning
        of each federated learning round.

        If mu > 0 and this method is not called, the optimizer will raise a ValueError
        during the optimization step.

        Args:
            old_weights: List of parameter tensors representing the global model weights.
                         Must match the order and structure of the model's parameters
                         being optimized (typically obtained by calling
                         [p.clone().detach() for p in model.parameters()]).
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

    def _validate_old_weights(self, mu, w_old):
        """Validate old weights for FedProx regularization.

        Args:
            mu: Proximal term coefficient
            w_old: Old weights reference

        Raises:
            ValueError: If mu > 0 and w_old is None
        """
        if mu > 0 and w_old is None:
            raise ValueError(
                "FedProx requires old weights to be set when mu > 0. "
                "Please call set_old_weights() before optimization step.",
            )

    def _apply_proximal_term(self, grad, param, w_old_p, mu):
        """Apply proximal term to gradient.

        Args:
            grad: Gradient
            param: Parameter
            w_old_p: Old weight parameter
            mu: Proximal term coefficient

        Returns:
            Modified gradient
        """
        return grad.add(param - w_old_p, alpha=mu)

    def _compute_adam_step(
        self,
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        step,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
    ):
        """Compute Adam optimization step.

        Args:
            param: Parameter
            grad: Gradient
            exp_avg: Exponential moving average
            exp_avg_sq: Exponential moving average squared
            max_exp_avg_sq: Maximum exponential moving average squared
            step: Step count
            amsgrad: Whether to use AMSGrad
            beta1: Beta1 coefficient
            beta2: Beta2 coefficient
            lr: Learning rate
            weight_decay: Weight decay
            eps: Epsilon value
        """
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)

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
        # Validate old weights for FedProx
        self._validate_old_weights(mu, w_old)

        # Apply proximal term when mu != 0
        apply_proximal = w_old is not None

        for i, param in enumerate(params):
            grad = grads[i]

            # Apply proximal term if needed
            if apply_proximal:
                w_old_p = w_old[i]
                grad = self._apply_proximal_term(grad, param, w_old_p, mu)

            # Apply Adam optimization steps
            self._compute_adam_step(
                param,
                grad,
                exp_avgs[i],
                exp_avg_sqs[i],
                max_exp_avg_sqs[i] if amsgrad else None,
                state_steps[i],
                amsgrad,
                beta1,
                beta2,
                lr,
                weight_decay,
                eps,
            )
