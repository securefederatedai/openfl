import numpy as np
from abc import ABC, abstractmethod


class AggregationFunctionInterface(ABC):
    """Interface for specifying aggregation function."""
    @abstractmethod
    def __call__(self, tensors: np.ndarray, **kwargs) -> np.ndarray:
        """
            Aggregate tensors.

            Args:
                tensors: array of `np.ndarray`s of tensors to aggregate.
                **kwargs: additional context passed to the function
        """
        raise NotImplementedError
