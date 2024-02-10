"""openfl.experimental.interface.aggregation_functions.fedcurv_weighted_average package."""

import numpy as np
from typing import Tuple

from .fedavg import FedAvg, weighted_average

class FedCurvAgg(FedAvg):
    """Fedcurv Weighted average aggregation.
    
    Applies weighted average aggregation to all tensors
    except Fisher matrices variables (u_t, v_t).
    These variables are summed without weights.

    FedCurv paper: https://arxiv.org/pdf/1910.07796.pdf
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the function."""
        rep = "FedCurv Aggregation Function"
        return rep

    def aggregate_models(self, model_weights, fisher_matrix_model_weights, collaborator_weights) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fedcurv weighted average."""
        # For Fisher variables, compute sum and for non Fisher elements, compute average
        return (np.sum(fisher_matrix_model_weights, axis=0), weighted_average(model_weights, collaborator_weights))
