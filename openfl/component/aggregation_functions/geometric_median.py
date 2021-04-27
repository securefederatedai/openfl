# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geometric median module."""

import numpy as np
from .fedavg import weighted_average


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
