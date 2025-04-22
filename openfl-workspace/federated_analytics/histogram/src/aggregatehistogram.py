# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Histogram module."""

import numpy as np

from openfl.interface.aggregation_functions.core import AggregationFunction


class AggregateHistogram(AggregationFunction):
    """Histogram aggregation."""

    def call(self, local_tensors, *_) -> np.ndarray:
        """
        Aggregates a list of local tensors into a single histogram.
        Args:
            local_tensors (list): A list of objects, each containing a tensor attribute which is a numpy array.
            *_: Additional arguments (unused).
        Returns:
            np.ndarray: The aggregated histogram as a numpy array. If the input list is empty, returns an empty numpy array.
        """

        if not local_tensors:
            return np.array([])

        agg_histogram = np.zeros_like(local_tensors[0].tensor)
        for local_tensor in local_tensors:
            agg_histogram += local_tensor.tensor
        return agg_histogram
