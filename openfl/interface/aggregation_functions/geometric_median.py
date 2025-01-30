# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Geometric median module."""

import numpy as np

from openfl.interface.aggregation_functions.core import AggregationFunction
from openfl.interface.aggregation_functions.weighted_average import weighted_average


def _geometric_median_objective(median, tensors, weights):
    """Compute geometric median objective.

    Args:
        median (np.ndarray): The median tensor.
        tensors (list): List of tensors.
        weights (list): List of weights corresponding to the tensors.

    Returns:
        float: The geometric median objective.
    """
    return sum([w * _l2dist(median, x) for w, x in zip(weights, tensors)])


def geometric_median(tensors, weights, maxiter=4, eps=1e-5, ftol=1e-6):
    """Compute geometric median of tensors with weights using Weiszfeld's
    Algorithm.

    Args:
        tensors (list): List of tensors.
        weights (list): List of weights corresponding to the tensors.
        maxiter (int, optional): Maximum number of iterations. Defaults to 4.
        eps (float, optional): Epsilon value for stability. Defaults to 1e-5.
        ftol (float, optional): Tolerance for convergence. Defaults to 1e-6.

    Returns:
        median (np.ndarray): The geometric median of the tensors.
    """
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
    """L2 distance between p1, p2, each of which is a list of nd-arrays.

    Args:
        p1 (np.ndarray): First tensor.
        p2 (np.ndarray): Second tensor.

    Returns:
        float: The L2 distance between the two tensors.
    """
    if p1.ndim != p2.ndim:
        raise RuntimeError("Tensor shapes should be equal")
    if p1.ndim < 2:
        return _l2dist(*[np.expand_dims(x, axis=0) for x in [p1, p2]])
    return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])


class GeometricMedian(AggregationFunction):
    """Geometric median aggregation."""

    def call(self, local_tensors, *_) -> np.ndarray:
        """Aggregate tensors.

        Args:
            local_tensors (list[openfl.utilities.LocalTensor]): List of local
                tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this
                    tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training
                        result. These tensors are passed to the aggregator
                        node after local learning.
                    - 'aggregated' indicates that tensor is a result of
                        aggregation. These tensors are sent to collaborators
                        for the next round.
                    - 'delta' indicates that value is a difference between
                        rounds for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            geometric_median (np.ndarray): aggregated tensor
        """
        tensors, weights = zip(*[(x.tensor, x.weight) for x in local_tensors])
        tensors, weights = np.array(tensors), np.array(weights)
        return geometric_median(tensors, weights)
