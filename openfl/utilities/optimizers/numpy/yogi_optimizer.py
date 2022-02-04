"""Adam optimizer module."""

import typing as tp

import numpy as np

from .adam_optimizer import Adam


class Yogi(Adam):
    """Yogi optimizer implementation."""

    def __init__(self,
                 params: tp.Dict[str, np.ndarray],
                 learning_rate: float = 0.01,
                 betas: tp.Tuple[float, float] = (0.9, 0.999),
                 initial_accumulator_value: float = 0.0,
                 epsilon: float = 1e-8,
                 ) -> None:
        """Initialize."""
        super().__init__(params,
                         learning_rate,
                         betas,
                         initial_accumulator_value,
                         epsilon)

    def step(self, gradients: tp.Dict[str, np.ndarray]) -> None:
        """
        Perform a single step for parameter update.

        Implement Yogi optimizer weights update rule.
        """
        for grad_name in gradients:
            grad = gradients[grad_name]

            if grad_name not in self.grads_first_moment:
                raise KeyError(f"Key {grad_name} doesn't exist in optimized parameters")

            self.grads_first_moment[grad_name] = (self.beta_1
                                                  * self.grads_first_moment[grad_name]
                                                  + (1.0 - self.beta_1) * grad)

            sign = np.sign(grad**2 - self.grads_second_moment[grad_name])
            self.grads_second_moment[grad_name] = (self.beta_2
                                                   * self.grads_second_moment[grad_name]
                                                   + (1.0 - self.beta_2) * sign * grad**2)

            grads_first_moment_normalized = (self.grads_first_moment[grad_name]
                                             / (1. - self.beta_1**(self.current_step + 1)))
            grads_second_moment_normalized = (self.grads_second_moment[grad_name]
                                              / (1. - self.beta_2**(self.current_step + 1)))

            self.params[grad_name] = (self.params[grad_name]
                                      - self.learning_rate
                                      * grads_first_moment_normalized
                                      / (np.sqrt(grads_second_moment_normalized) + self.epsilon))

        self.current_step += 1
