# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Adam optimizer module."""

from typing import Dict, Optional, Tuple

import numpy as np

from openfl.utilities.optimizers.numpy.base_optimizer import Optimizer


class NumPyAdam(Optimizer):
    """Adam optimizer implementation.

    Implements the Adam optimization algorithm using NumPy.
    Adam is an algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order
    moments.

    Original paper: https://openreview.net/forum?id=ryQu7f-RZ

    Attributes:
        params (dict, optional): Parameters to be stored for optimization.
        model_interface: Model interface instance to provide parameters.
        learning_rate (float): Tuning parameter that determines the step size
            at each iteration.
        betas (tuple): Coefficients used for computing running averages of
            gradient and its square.
        initial_accumulator_value (float): Initial value for gradients and
            squared gradients.
        epsilon (float): Value for computational stability.
    """

    def __init__(
        self,
        *,
        params: Optional[Dict[str, np.ndarray]] = None,
        model_interface=None,
        learning_rate: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        initial_accumulator_value: float = 0.0,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize the Adam optimizer.

        Args:
            params (dict, optional): Parameters to be stored for optimization.
                Defaults to None.
            model_interface: Model interface instance to provide parameters.
                Defaults to None.
            learning_rate (float, optional): Tuning parameter that determines
                the step size at each iteration. Defaults to 0.01.
            betas (tuple, optional): Coefficients used for computing running
                averages of gradient and its square. Defaults to (0.9, 0.999).
            initial_accumulator_value (float, optional): Initial value for
                gradients and squared gradients. Defaults to 0.0.
            epsilon (float, optional): Value for computational stability.
                Defaults to 1e-8.

        Raises:
            ValueError: If both params and model_interface are None.
            ValueError: If learning_rate is less than 0.
            ValueError: If betas[0] is not in [0, 1).
            ValueError: If betas[1] is not in [0, 1).
            ValueError: If initial_accumulator_value is less than 0.
            ValueError: If epsilon is less than or equal to 0.
        """
        super().__init__()

        if model_interface is None and params is None:
            raise ValueError("Should provide one of the params or model_interface")

        if learning_rate < 0:
            raise ValueError(f"Invalid learning rate: {learning_rate}. Learning rate must be >= 0.")
        if not 0.0 <= betas[0] < 1:
            raise ValueError(f"Invalid betas[0] value: {betas[0]}. betas[0] must be in [0, 1).")
        if not 0.0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas[1] value: {betas[1]}. betas[1] must be in [0, 1).")
        if initial_accumulator_value < 0:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}. \
                Initial accumulator value must be >= 0."
            )
        if epsilon <= 0:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Epsilon avalue must be > 0.")

        self.params = params

        if params is None and model_interface is not None:
            self._set_params_from_model(model_interface)

        self.learning_rate = learning_rate
        self.beta_1, self.beta_2 = betas
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon
        self.current_step = dict.fromkeys(self.params, 0)

        self.grads_first_moment, self.grads_second_moment = {}, {}

        for param_name in self.params:
            self.grads_first_moment[param_name] = np.full_like(
                self.params[param_name], self.initial_accumulator_value
            )
            self.grads_second_moment[param_name] = np.full_like(
                self.params[param_name], self.initial_accumulator_value
            )

    def _update_first_moment(self, grad_name: str, grad: np.ndarray) -> None:
        """
        Update gradients first moment.

        Args:
            grad_name (str): The name of the gradient.
            grad (np.ndarray): The gradient values.
        """
        self.grads_first_moment[grad_name] = self.beta_1 * self.grads_first_moment[grad_name] + (
            (1.0 - self.beta_1) * grad
        )

    def _update_second_moment(self, grad_name: str, grad: np.ndarray) -> None:
        """
        Update gradients second moment.

        Args:
            grad_name (str): The name of the gradient.
            grad (np.ndarray): The gradient values.
        """
        self.grads_second_moment[grad_name] = self.beta_2 * self.grads_second_moment[grad_name] + (
            (1.0 - self.beta_2) * grad**2
        )

    def step(self, gradients: Dict[str, np.ndarray]) -> None:
        """Perform a single step for parameter update.

        Implement Adam optimizer weights update rule.

        Args:
            gradients (dict): Partial derivatives with respect to optimized
                parameters.

        Raises:
            KeyError: If a key in gradients does not exist in optimized
                parameters.
        """
        for grad_name in gradients:
            if grad_name not in self.grads_first_moment:
                raise KeyError(f"Key {grad_name} doesn't exist in optimized parameters")

            grad = gradients[grad_name]

            self._update_first_moment(grad_name, grad)
            self._update_second_moment(grad_name, grad)

            t = self.current_step[grad_name] + 1
            mean = self.grads_first_moment[grad_name]
            var = self.grads_second_moment[grad_name]

            grads_first_moment_normalized = mean / (1.0 - self.beta_1**t)
            grads_second_moment_normalized = var / (1.0 - self.beta_2**t)

            # Make an update for a group of parameters
            self.params[grad_name] -= (
                self.learning_rate
                * grads_first_moment_normalized
                / (np.sqrt(grads_second_moment_normalized) + self.epsilon)
            )

            self.current_step[grad_name] += 1
