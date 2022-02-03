"""Base abstract optimizer class module."""
import abc


class Optimizer(abc.ABC):
    """Base abstract optimizer class."""

    @abc.abstractmethod
    def step(self):
        """Perform a single step for parameter update."""
        pass
