# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Adam optimizer module."""

from typing import Dict, Optional, Tuple

import numpy as np

from openfl.utilities.optimizers.numpy.adam_optimizer import NumPyAdam


class NumPyYogi(NumPyAdam):
    """Yogi optimizer implementation.

    Implements the Yogi optimization algorithm using NumPy.
    Yogi is an algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order
    moments. It is a variant of Adam and it is more robust to large learning
    rates.

    Original paper:
    https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization

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
        """Initialize the Yogi optimizer.

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
        """
        super().__init__(
            params=params,
            model_interface=model_interface,
            learning_rate=learning_rate,
            betas=betas,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon,
        )

    def _update_second_moment(self, grad_name: str, grad: np.ndarray) -> None:
        """Override second moment update rule for Yogi optimization updates.

        Args:
            grad_name (str): The name of the gradient.
            grad (np.ndarray): The gradient values.
        """
        sign = np.sign(grad**2 - self.grads_second_moment[grad_name])
        self.grads_second_moment[grad_name] = (
            self.beta_2 * self.grads_second_moment[grad_name] + (1.0 - self.beta_2) * sign * grad**2
        )

    def step(self, gradients: Dict[str, np.ndarray]) -> None:
        """Perform a single step for parameter update.

        Implement Yogi optimizer weights update rule.

        Args:
            gradients (dict): Partial derivatives with respect to optimized
                parameters.
        """
        super().step(gradients)
