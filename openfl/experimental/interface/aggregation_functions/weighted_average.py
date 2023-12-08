"""openfl.experimental.interface.aggregation_functions.weighted_average package."""

import numpy as np

from .interface import AggregationFunction


def weighted_average(tensors, weights):
    """Compute weighted average"""
    return np.average(tensors, weights=weights, axis=0)



class WeightedAverage(AggregationFunction):
    """Weighted average aggregation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the function."""
        rep = "FedAvg"
        return rep


    def aggregate_models(self, model_weights, collaborator_weights):
        """Compute weighted average."""
        return weighted_average(model_weights, collaborator_weights)


    def aggregate_metrics(self, **kwargs):
        """Aggregate loss"""
        pass
 
