# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Histogram module."""

import numpy as np

from openfl.interface.aggregation_functions.core import AggregationFunction


class Histogram(AggregationFunction):
    """Histogram aggregation."""

    def call(self, local_tensors, *_) -> np.ndarray:
        print("Histogram called")
        agg_hist = {}
        for local_tensor_key, local_tensor in local_tensors.items():
            tensor_name, origin, fl_round, report, tags = local_tensor_key.split(':')
            # if tensor_name not in agg_hist:
            #     agg_hist[tensor_name] = np.zeros_like(local_tensor)
            # agg_hist[tensor_name] += local_tensor

            if tensor_name not in agg_hist:
                agg_hist[tensor_name] = np.zeros_like(local_tensor)
            agg_hist[tensor_name] += local_tensor

        return agg_hist


        # print("local_tensor_key", local_tensor_key)
        #     print(local_tensor)
        #     if 'sepal length (cm)' in local_tensor_key:
        #         length_agg_hist += local_tensor
        #     elif 'sepal width (cm)' in local_tensor_key:
        #         width_agg_hist += local_tensor
        # return np.concatenate((["Length:"], length_agg_hist, ["Width:"], width_agg_hist))
    
