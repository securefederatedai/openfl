# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geometric median module."""

from .interface import AggregationFunctionInterface
import numpy as np
from .weighted_average import weighted_average


def _geometric_median_objective(median, tensors, weights):
    """Compute geometric median objective."""
    return sum([w * _l2dist(median, x) for w, x in zip(weights, tensors)])


def geometric_median(tensors, weights, maxiter=4, eps=1e-5, ftol=1e-6):
    """Compute geometric median of tensors with weights using Weiszfeld's Algorithm."""
    weights = np.asarray(weights) / sum(weights)
    median = weighted_average(tensors, weights)
    num_oracle_calls = 1

    obj_val = _geometric_median_objective(median, tensors, weights)

    for _ in range(maxiter):
        prev_obj_val = obj_val
        weights = np.asarray([w / max(eps, _l2dist(median, x)) for w, x in zip(weights, tensors)])
        weights = weights / weights.sum()
        median = weighted_average(tensors, weights)
        num_oracle_calls += 1
        obj_val = _geometric_median_objective(median, tensors, weights)
        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break
    return median


def _l2dist(p1, p2):
    """L2 distance between p1, p2, each of which is a list of nd-arrays."""
    if p1.ndim != p2.ndim:
        raise RuntimeError('Tensor shapes should be equal')
    if p1.ndim < 2:
        return _l2dist(*[np.expand_dims(x, axis=0) for x in [p1, p2]])
    return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])


class GeometricMedian(AggregationFunctionInterface):
    """Geometric median aggregation."""

    def call(self, agg_tensor_dict, weights, *_) -> np.ndarray:
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
        return geometric_median(tensors, weights)
