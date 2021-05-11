# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Median module."""

from .interface import AggregationFunctionInterface
import numpy as np


class Median(AggregationFunctionInterface):
    """Median aggregation."""

    def call(self, local_tensors, *_):
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
        tensors = np.array([x.tensor for x in local_tensors])
        return np.median(tensors, axis=0)
