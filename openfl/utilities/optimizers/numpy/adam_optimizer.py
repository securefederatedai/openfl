"""Adam optimizer module."""

import typing as tp

import numpy as np

from .base_optimizer import Optimizer


class Adam(Optimizer):
    """Adagrad optimizer implementation.

    Original paper: https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 params: tp.Dict[str, np.ndarray],
                 learning_rate: float = 0.01,
                 betas: tp.Tuple[float, float] = (0.9, 0.999),
                 initial_accumulator_value: float = 0.0,
                 epsilon: float = 1e-8,
                 ) -> None:
        """Initialize.

        Args:
            params: Parameters to be stored for optimization.
            learning_rate: Tuning parameter that determines
                the step size at each iteration.
            betas: Coefficients used for computing running
                averages of gradient and its square.
            initial_accumulator_value: Initial value for gradients
                and squared gradients.
            epsilon: Value for computational stability.
        """
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

    def _update_first_moment(self, grad_name: str, grad: np.ndarray) -> None:
        """Update gradients first moment."""
        self.grads_first_moment[grad_name] = (self.beta_1
                                              * self.grads_first_moment[grad_name]
                                              + (1.0 - self.beta_1) * grad)

    def _update_second_moment(self, grad_name: str, grad: np.ndarray) -> None:
        """Update gradients second moment."""
        self.grads_second_moment[grad_name] = (self.beta_2
                                               * self.grads_second_moment[grad_name]
                                               + (1.0 - self.beta_2) * grad**2)

    def step(self, gradients: tp.Dict[str, np.ndarray]) -> None:
        """
        Perform a single step for parameter update.

        Implement Adam optimizer weights update rule.

        Args:
            gradients: Partial derivatives with respect to optimized parameters.
        """
        for grad_name in gradients:
            grad = gradients[grad_name]

            if grad_name not in self.grads_first_moment:
                raise KeyError(f"Key {grad_name} doesn't exist in optimized parameters")

            self._update_first_moment(grad_name, grad)
            self._update_second_moment(grad_name, grad)

            grads_first_moment_normalized = (self.grads_first_moment[grad_name]
                                             / (1. - self.beta_1**(self.current_step + 1)))
            grads_second_moment_normalized = (self.grads_second_moment[grad_name]
                                              / (1. - self.beta_2**(self.current_step + 1)))

            # Make an update for a group of parameters
            self.params[grad_name] = (self.params[grad_name]
                                      - self.learning_rate
                                      * grads_first_moment_normalized
                                      / (np.sqrt(grads_second_moment_normalized) + self.epsilon))

        self.current_step += 1
