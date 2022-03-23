# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adagrad optimizer module."""

from typing import Dict
from typing import Optional

import numpy as np

from .base_optimizer import Optimizer


class NumPyAdagrad(Optimizer):
    """Adagrad optimizer implementation.

    Original paper: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        *,
        params: Optional[Dict[str, np.ndarray]] = None,
        model_interface=None,
        learning_rate: float = 0.01,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-10,
    ) -> None:
        """Initialize.

        Args:
            params: Parameters to be stored for optimization.
            model_interface: Model interface instance to provide parameters.
            learning_rate: Tuning parameter that determines
                the step size at each iteration.
            initial_accumulator_value: Initial value for squared gradients.
            epsilon: Value for computational stability.
        """
        super().__init__()

        if model_interface is None and params is None:
            raise ValueError('Should provide one of the params or model_interface')

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

        self.params = params

        if params is None and model_interface is not None:
            self._set_params_from_model(model_interface)

        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

        self.grads_squared = {}
        for param_name in self.params:
            self.grads_squared[param_name] = np.full_like(self.params[param_name],
                                                          self.initial_accumulator_value)

    def _update_param(self, grad_name: str, grad: np.ndarray) -> None:
        """Update papams by given gradients."""
        self.params[grad_name] -= (self.learning_rate * grad
                                   / (np.sqrt(self.grads_squared[grad_name]) + self.epsilon))

    def step(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Perform a single step for parameter update.

        Implement Adagrad optimizer weights update rule.

        Args:
            gradients: Partial derivatives with respect to optimized parameters.
        """
        for grad_name in gradients:
            if grad_name not in self.grads_squared:
                raise KeyError(f"Key {grad_name} doesn't exist in optimized parameters")

            grad = gradients[grad_name]
            self.grads_squared[grad_name] = self.grads_squared[grad_name] + grad**2
            self._update_param(grad_name, grad)
