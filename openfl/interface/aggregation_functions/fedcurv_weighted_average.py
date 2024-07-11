# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""FedCurv Aggregation function module."""
import numpy as np

from openfl.interface.aggregation_functions.weighted_average import WeightedAverage


class FedCurvWeightedAverage(WeightedAverage):
    """Aggregation function of FedCurv algorithm.

    Applies weighted average aggregation to all tensors
    except Fisher matrices variables (u_t, v_t).
    These variables are summed without weights.

    FedCurv paper: https://arxiv.org/pdf/1910.07796.pdf
    """

    def call(self, local_tensors, tensor_db, tensor_name, fl_round, tags):
        """Apply aggregation."""
        if tensor_name.endswith("_u") or tensor_name.endswith("_v") or tensor_name.endswith("_w"):
            tensors = [local_tensor.tensor for local_tensor in local_tensors]
            agg_result = np.sum(tensors, axis=0)
            return agg_result
        return super().call(local_tensors)
