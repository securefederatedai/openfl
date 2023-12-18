"""openfl.experimental.interface.aggregation_functions.weighted_average package."""

import numpy as np
from typing import List

from .interface import AggregationFunction


def weighted_average(tensors, weights):
    """Compute weighted average"""
    return np.average(tensors, weights=weights, axis=0)


class FedAvg(AggregationFunction):
    """Federated average aggregation.
    
    FedAvg paper: https://arxiv.org/pdf/1602.05629.pdf
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the function."""
        rep = "FedAvg Aggregation Function"
        return rep


    def aggregate_models(self, model_weights, collaborator_weights) -> np.ndarray:
        """Compute fed avg across models.""" 
        return weighted_average(model_weights, collaborator_weights)


    def aggregate_metrics(self, metrics) -> List[np.ndarray]:
        """Aggregate metrics like loss and accuracy"""
        agg_metrics = []
        for metric in metrics:
            agg_metrics.append(weighted_average(metric, weights=None))
        return agg_metrics
 
