"""Base abstract optimizer class module."""
import abc
from typing import Dict

from numpy import ndarray


class Optimizer(abc.ABC):
    """Base abstract optimizer class."""

    @abc.abstractmethod
    def step(self, gradients: Dict[str, ndarray]) -> None:
        """Perform a single step for parameter update.

        Args:
            gradients: Partial derivatives with respect to optimized parameters.
        """
        pass
