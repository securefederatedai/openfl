# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Federated averaging module."""

from .interface import AggregationFunctionInterface
import numpy as np


def weighted_average(tensors, weights):
    """Compute average."""
    return np.average(tensors, weights=weights, axis=0)


class WeightedAverage(AggregationFunctionInterface):
    """Weighted average aggregation."""

    def call(self, agg_tensor_dict, weights, *_):
        """Aggregate tensors.

        Args:
            agg_tensor_dict: Dict of (collaborator name, tensor) pairs to aggregate.
            weights: array of floats representing data partition (sum up to 1)
            db_iterator: iterator over history of all tensors.
                Columns: ['tensor_name', 'round', 'tags', 'nparray']
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        """
        tensors = np.array(list(agg_tensor_dict.values()))
        return weighted_average(tensors, weights)
