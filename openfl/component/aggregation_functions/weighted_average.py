# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Federated averaging module."""

from .interface import AggregationFunctionInterface
import numpy as np


def weighted_average(tensors, weights):
    """Compute simple weighted average (FedAvg operation)."""
    return np.average(tensors, weights=weights, axis=0)


class WeightedAverage(AggregationFunctionInterface):
    """Weighted average aggregation."""

    def __call__(self, tensors: np.ndarray, **kwargs) -> np.ndarray:
        """Aggregate tensors.

        Args:
            tensors: array of `np.ndarray`s of tensors to aggregate.
            **kwargs: additional context passed to the function
        """
        return weighted_average(tensors, kwargs['weights'])
