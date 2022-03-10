# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""FedCurv Aggregation function module."""
from typing import Any
from typing import List

import numpy as np

from .weighted_average import WeightedAverage


class FedCurvWeightedAverage(WeightedAverage):
    """Aggregation function of FedCurv algorithm.

    Applies weighted average aggregation to all tensors
    except Fisher matrices variables (u_t, v_t).
    These variables are summed without weights.

    FedCurv paper: https://arxiv.org/pdf/1910.07796.pdf
    """

    def call(self, local_tensors: List, db_iterator: Any,
             tensor_name: str, fl_round: Any, tags: Any) -> np.ndarray:
        """Apply aggregation."""
        if (
            tensor_name.endswith('_u')
            or tensor_name.endswith('_v')
            or tensor_name.endswith('_w')
        ):
            tensors = [local_tensor.tensor for local_tensor in local_tensors]
            agg_result = np.sum(tensors, axis=0)
            return agg_result
        return super().call(local_tensors)
