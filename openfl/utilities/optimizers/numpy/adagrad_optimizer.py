# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Adagrad optimizer module."""

from typing import Dict, Optional

import numpy as np

from openfl.utilities.optimizers.numpy.base_optimizer import Optimizer


class NumPyAdagrad(Optimizer):
    """Adagrad optimizer implementation.

    Implements the Adagrad optimization algorithm using NumPy. Adagrad is an
    algorithm for gradient-based optimization that adapts the learning rate to
    the parameters, performing smaller updates for parameters associated with
    frequently occurring features, and larger updates for parameters
    associated with infrequent features.

    Original paper: http://jmlr.org/papers/v12/duchi11a.html

    Attributes:
        params (dict, optional): Parameters to be stored for optimization.
        model_interface: Model interface instance to provide parameters.
        learning_rate (float): Tuning parameter that determines the step size
            at each iteration.
        initial_accumulator_value (float): Initial value for squared gradients.
        epsilon (float): Value for computational stability.
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
        """Initialize the Adagrad optimizer.

        Args:
            params (dict, optional): Parameters to be stored for optimization.
                Defaults to None.
            model_interface: Model interface instance to provide parameters.
                Defaults to None.
            learning_rate (float, optional): Tuning parameter that determines
                the step size at each iteration. Defaults to 0.01.
            initial_accumulator_value (float, optional): Initial value for
                squared gradients. Defaults to 0.1.
            epsilon (float, optional): Value for computational stability.
                Defaults to 1e-10.

        Raises:
            ValueError: If both params and model_interface are None.
            ValueError: If learning_rate is less than 0.
            ValueError: If initial_accumulator_value is less than 0.
            ValueError: If epsilon is less than or equal to 0.
        """
        super().__init__()

        if model_interface is None and params is None:
            raise ValueError("Should provide one of the params or model_interface")

        if learning_rate < 0:
            raise ValueError(f"Invalid learning rate: {learning_rate}. Learning rate must be >= 0.")
        if initial_accumulator_value < 0:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}."
                "Initial accumulator value must be >= 0."
            )
        if epsilon <= 0:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Epsilon avalue must be > 0.")

        self.params = params

        if params is None and model_interface is not None:
            self._set_params_from_model(model_interface)

        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

        self.grads_squared = {}
        for param_name in self.params:
            self.grads_squared[param_name] = np.full_like(
                self.params[param_name], self.initial_accumulator_value
            )

    def _update_param(self, grad_name: str, grad: np.ndarray) -> None:
        """
        Update papams by given gradients.

        Args:
            grad_name (str): The name of the gradient.
            grad (np.ndarray): The gradient values.
        """
        self.params[grad_name] -= (
            self.learning_rate * grad / (np.sqrt(self.grads_squared[grad_name]) + self.epsilon)
        )

    def step(self, gradients: Dict[str, np.ndarray]) -> None:
        """Perform a single step for parameter update.

        Implement Adagrad optimizer weights update rule.

        Args:
            gradients (dict): Partial derivatives with respect to optimized
                parameters.

        Raises:
            KeyError: If a key in gradients does not exist in optimized
                parameters.
        """
        for grad_name in gradients:
            if grad_name not in self.grads_squared:
                raise KeyError(f"Key {grad_name} doesn't exist in optimized parameters")

            grad = gradients[grad_name]
            self.grads_squared[grad_name] = self.grads_squared[grad_name] + grad**2
            self._update_param(grad_name, grad)
