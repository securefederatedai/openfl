# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Histogram module."""

import numpy as np

# def exp():
#     a = [1,1,1,2,2,3,4,5]
#     print(np.histogram(a))

# exp()

from openfl.interface.aggregation_functions.core import AggregationFunction


class Histogram(AggregationFunction):
    """Histogram aggregation."""

    def call(self, local_tensors, *_) -> np.ndarray:
        print("Histogram called")
        length_agg_hist = 0
        width_agg_hist = 0



        for local_tensor_key, local_tensor in local_tensors.items():
            tensor_name, origin, fl_round, report, tags = local_tensor_key.split(':')
            # if tensor_name not in agg_hist:
            #     agg_hist[tensor_name] = np.zeros_like(local_tensor)
            # agg_hist[tensor_name] += local_tensor
            print("local_tensor_key", local_tensor_key)
            print(local_tensor)
            # length_agg_hist += val[0]
            # width_agg_hist += val[1]
        return {"a": [], "b": []}
        # return np.concatenate((["Length:"], length_agg_hist, ["Width:"], width_agg_hist))
    
