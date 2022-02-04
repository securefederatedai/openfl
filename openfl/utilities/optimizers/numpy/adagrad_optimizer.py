"""Adagrad optimizer module."""

import typing as tp

import numpy as np

from .base_optimizer import Optimizer


class Adagrad(Optimizer):
    """Adagrad optimizer implementation."""

    def __init__(self,
                 params: tp.Dict[str, np.ndarray],
                 learning_rate: float = 0.01,
                 initial_accumulator_value: float = 0.1,
                 epsilon: float = 1e-10,
                 ) -> None:
        """Initialize."""
        super().__init__()

        if learning_rate < 0:
            raise ValueError(f'Invalid learning rate: {learning_rate}')
        if initial_accumulator_value < 0:
            raise ValueError(
                f'Invalid initial_accumulator_value value: {initial_accumulator_value}')
        if epsilon < 0:
            raise ValueError(
                f'Invalid epsilon value: {epsilon}')

        self.params = params
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

        self.grads_squared = {}
        for param_name in self.params:
            self.grads_squared[param_name] = np.full_like(self.params[param_name],
                                                          self.initial_accumulator_value)

    def step(self, gradients: tp.Dict[str, np.ndarray]) -> None:
        """
        Perform a single step for parameter update.

        Implement Adagrad optimizer weights update rule.
        TODO: math latex symbols
        """
        for grad_name in gradients:
            grad = gradients[grad_name]

            if grad_name not in self.grads_squared:
                raise KeyError(f"Key {grad_name} doesn't exist in optimized parameters")

            self.grads_squared[grad_name] = self.grads_squared[grad_name] + grad**2

            self.params[grad_name] = (self.params[grad_name] - self.learning_rate * grad
                                      / (np.sqrt(self.grads_squared[grad_name]) + self.epsilon))
