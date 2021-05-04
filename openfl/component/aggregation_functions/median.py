# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Median module."""

from .interface import AggregationFunctionInterface
import numpy as np


class Median(AggregationFunctionInterface):
    """Median aggregation."""

    def call(self, tensors, *_):
        """Aggregate tensors.

        Args:
            tensors: array of `np.ndarray`s of tensors to aggregate.
            weights: array of floats representing data partition (sum up to 1)
            db_iterator: iterator over history of aggregated versions of this tensor
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        """
        return np.median(tensors, axis=0)
