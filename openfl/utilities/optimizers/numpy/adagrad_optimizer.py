# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adagrad optimizer module."""

from typing import Dict

import numpy as np

from .base_optimizer import Optimizer


class NumPyAdagrad(Optimizer):
    """Adagrad optimizer implementation.

    Original paper: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.01,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-10,
    ) -> None:
        """Initialize.

        Args:
            learning_rate: Tuning parameter that determines
                the step size at each iteration.
            initial_accumulator_value: Initial value for squared gradients.
            epsilon: Value for computational stability.
        """
        super().__init__()

        if learning_rate < 0:
            raise ValueError(
                f'Invalid learning rate: {learning_rate}. Learning rate must be >= 0.')
        if initial_accumulator_value < 0:
            raise ValueError(
                f'Invalid initial_accumulator_value value: {initial_accumulator_value}.'
                'Initial accumulator value must be >= 0.')
        if epsilon <= 0:
            raise ValueError(
                f'Invalid epsilon value: {epsilon}. Epsilon avalue must be > 0.')

        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

        self.grads_squared = {}

    def _update_param(self, param: np.ndarray, grad_name: str, grad: np.ndarray) -> np.ndarray:
        """Update param by given gradient."""
        return param - (self.learning_rate * grad
                        / (np.sqrt(self.grads_squared[grad_name]) + self.epsilon))

    def step(
        self,
        params: Dict[str, np.ndarray],
        gradients: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Perform a single step for parameter update.

        Implement Adagrad optimizer weights update rule.

        Args:
            params: Optimized parameters
            gradients: Partial derivatives with respect to optimized parameters.
        """
        result = {}
        for grad_name in gradients:
            grad = gradients[grad_name]
            param = params[grad_name]
            if grad_name not in self.grads_squared:
                self.grads_squared[grad_name] = np.full_like(grad, self.initial_accumulator_value)

            self.grads_squared[grad_name] = self.grads_squared[grad_name] + grad**2

            updated_param = self._update_param(param, grad_name, grad)
            result[grad_name] = updated_param
        return result
