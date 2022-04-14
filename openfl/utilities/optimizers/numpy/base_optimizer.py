# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base abstract optimizer class module."""
import abc
from typing import Dict

from numpy import ndarray


class Optimizer(abc.ABC):
    """Base abstract optimizer class."""

    @abc.abstractmethod
    def step(
        self,
        params: Dict[str, ndarray],
        gradients: Dict[str, ndarray]
    ) -> Dict[str, ndarray]:
        """Perform a single step for parameter update.

        Args:
            params: Optimized parameters.
            gradients: Partial derivatives with respect to optimized parameters.

        Returns:
            Updated parameters.
        """
        pass
