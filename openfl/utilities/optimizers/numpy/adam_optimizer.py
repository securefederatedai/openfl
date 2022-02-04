"""Adam optimizer module."""

import typing as tp

import numpy as np

from .base_optimizer import Optimizer


class Adam(Optimizer):
    """Adagrad optimizer implementation."""

    def __init__(self,
                 params: tp.Dict[str, np.ndarray],
                 learning_rate: float = 0.01,
                 betas: tp.Tuple[float, float] = (0.9, 0.999),
                 initial_accumulator_value: float = 0.0,
                 epsilon: float = 1e-8,
                 ) -> None:
        """Initialize."""
        super().__init__()

        if learning_rate < 0:
            raise ValueError(f'Invalid learning rate: {learning_rate}')
        if not 0.0 <= betas[0] < 1:
            raise ValueError(
                f'Invalid betas[0] value: {betas[0]}')
        if not 0.0 <= betas[1] < 1:
            raise ValueError(
                f'Invalid betas[1] value: {betas[1]}')
        if initial_accumulator_value < 0:
            raise ValueError(
                f'Invalid initial_accumulator_value value: {initial_accumulator_value}')
        if epsilon < 0:
            raise ValueError(
                f'Invalid epsilon value: {epsilon}')

        self.params = params
        self.learning_rate = learning_rate
        self.beta_1, self.beta_2 = betas
        self.epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value
        self.current_step = 0

        self.grads_first_moment, self.grads_second_moment = {}, {}

        for param_name in self.params:
            self.grads_first_moment[param_name] = np.full_like(self.params[param_name],
                                                               self.initial_accumulator_value)
            self.grads_second_moment[param_name] = np.full_like(self.params[param_name],
                                                                self.initial_accumulator_value)

    def step(self, gradients: tp.Dict[str, np.ndarray]) -> None:
        """
        Perform a single step for parameter update.

        Implement Adam optimizer weights update rule.
        """
        for grad_name in gradients:
            grad = gradients[grad_name]

            if grad_name not in self.grads_first_moment:
                raise KeyError(f"Key {grad_name} doesn't exist in optimized parameters")

            self.grads_first_moment[grad_name] = (self.beta_1
                                                  * self.grads_first_moment[grad_name]
                                                  + (1.0 - self.beta_1) * grad)
            self.grads_second_moment[grad_name] = (self.beta_2
                                                   * self.grads_second_moment[grad_name]
                                                   + (1.0 - self.beta_2) * grad**2)

            grads_first_moment_normalized = (self.grads_first_moment[grad_name]
                                             / (1. - self.beta_1**(self.current_step + 1)))
            grads_second_moment_normalized = (self.grads_second_moment[grad_name]
                                              / (1. - self.beta_2**(self.current_step + 1)))

            self.params[grad_name] = (self.params[grad_name]
                                      - self.learning_rate
                                      * grads_first_moment_normalized
                                      / (np.sqrt(grads_second_moment_normalized) + self.epsilon))

        self.current_step += 1
