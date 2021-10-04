import numpy as np
from openfl.component.aggregation_functions import WeightedAverage


class FedCurvWeightedAverage(WeightedAverage):
    """Aggregation function of FedCurv algorithm.

    Applies weighted average aggregation to all tensors except Fisher matrices variables (u_t, v_t).
    These variables are summed without weights.
    
    FedCurv paper: https://arxiv.org/pdf/1910.07796.pdf
    """
    def call(self, local_tensors, db_iterator, tensor_name, fl_round, tags):
        if (
            tensor_name.endswith('_u')
            or tensor_name.endswith('_v')
        ):
            return np.sum(local_tensors)
        return super().call(local_tensors)
