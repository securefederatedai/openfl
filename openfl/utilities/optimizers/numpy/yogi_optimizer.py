# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adam optimizer module."""

from typing import Dict, Optional, Tuple

import numpy as np

from openfl.utilities.optimizers.numpy.adam_optimizer import NumPyAdam


class NumPyYogi(NumPyAdam):
    """Yogi optimizer implementation.

    Original paper:
    https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
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
        """Initialize.

        Args:
            params: Parameters to be stored for optimization.
            model_interface: Model interface instance to provide parameters.
            learning_rate: Tuning parameter that determines
                the step size at each iteration.
            betas: Coefficients used for computing running
                averages of gradient and its square.
            initial_accumulator_value: Initial value for gradients
                and squared gradients.
            epsilon: Value for computational stability.
        """
        super().__init__(params=params,
                         model_interface=model_interface,
                         learning_rate=learning_rate,
                         betas=betas,
                         initial_accumulator_value=initial_accumulator_value,
                         epsilon=epsilon)

    def _update_second_moment(self, grad_name: str, grad: np.ndarray) -> None:
        """Override second moment update rule for Yogi optimization updates."""
        sign = np.sign(grad**2 - self.grads_second_moment[grad_name])
        self.grads_second_moment[grad_name] = (self.beta_2
                                               * self.grads_second_moment[grad_name]
                                               + (1.0 - self.beta_2) * sign * grad**2)

    def step(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Perform a single step for parameter update.

        Implement Yogi optimizer weights update rule.

        Args:
            gradients: Partial derivatives with respect to optimized parameters.
        """
        super().step(gradients)
