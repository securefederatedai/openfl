"""openfl.experimental.interface.aggregation_functions.fedprox package."""
import numpy as np
from typing import List

from .fedavg import FedAvg, weighted_average


class FedProxAgg(FedAvg):
    """FedProx aggregation.
    
    A representation of FedAvg with inclusion of the
    proximal term for heterogeneous data distributions
    
    FedProx paper: https://arxiv.org/pdf/1812.06127.pdf
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the function."""
        rep = "FedProx Aggregation Function"
        return rep

    def aggregate_metrics(self, metrics, weights) -> List[np.ndarray]:
        """Weighted average of loss and accuracy metrics"""
        agg_metrics = []
        for idx,metric in enumerate(metrics):
            agg_metrics.append(weighted_average(metric, weights=weights[idx]))
        return agg_metrics