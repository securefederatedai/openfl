# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Histogram module."""

import numpy as np

from openfl.interface.aggregation_functions.core import AggregationFunction


class AggregateHistogram(AggregationFunction):
    """Histogram aggregation."""

    def call(self, local_tensors, *_) -> np.ndarray:
        """
        Aggregates a list of local histograms into a single global histogram.

        This method takes a list of objects, each containing a `tensor` attribute
        (which represents a local histogram as a numpy array), and combines them
        into a single aggregated histogram. A histogram is a representation of
        the distribution of data, typically used to count occurrences of values
        within specified ranges (bins). Aggregating histograms involves summing
        up the corresponding bins across all local histograms to produce a global
        view of the data distribution.

            local_tensors (list): A list of objects, where each object has a
                `tensor` attribute that is a numpy array representing a local
                histogram.

            np.ndarray: The aggregated histogram as a numpy array. Each bin in
            the resulting histogram is the sum of the corresponding bins from
            all input histograms. If the input list is empty, raises a
            ValueError indicating that the result is empty.

        Raises:
            ValueError: If the input list `local_tensors` is empty, indicating
            that there are no histograms to aggregate.
        """

        if not local_tensors:
            raise ValueError("Histogram result is empty.")

        agg_histogram = np.zeros_like(local_tensors[0].tensor)
        for local_tensor in local_tensors:
            agg_histogram += local_tensor.tensor
        return agg_histogram
