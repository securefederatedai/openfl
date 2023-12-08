"""openfl.experimental.interface.aggregation_functions.fedcurv_weighted_average package."""

import numpy as np

from .interface import AggregationFunction
from openfl.interface.aggregation_functions.weighted_average import weighted_average

class FedcurvWeightedAverage(WeightedAverage):
    """Fedcurv Weighted average aggregation."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def aggregate_models(self, model_weights, collaborator_weights, state_keys):
        """Compute weighted average."""
        return weighted_average(model_weights, collaborator_weights)
    
    def __call__(self, tensors_list, weights_list, state_dict_keys_list) -> np.ndarray:        
        final_weights_list = []
        for key,val in dict_.items():
            if (key[-2:] == '_u' or key[-2:] == '_v' or key[-2:] == '_w'):
                final_weights_list.append(np.sum())
                continue
            final_weights_list.append(np.average())