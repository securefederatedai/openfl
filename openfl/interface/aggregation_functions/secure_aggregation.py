# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Federated averaging with secure aggregation module."""

import numpy as np

from openfl.interface.aggregation_functions.core import AggregationFunction


class SecureAggregation(AggregationFunction):
    """FedAvg with secure aggregation"""

    def call(self, local_tensors, db_iterator, *_) -> np.ndarray:
        """Aggregate tensors.

        Args:
            local_tensors (list[openfl.utilities.LocalTensor]): List of local
                tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight','fc2.bias'.
                - 'round': 0-based number of round corresponding to this
                    tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training
                        result.
                        These tensors are passed to the aggregator node after
                        local learning.
                    - 'aggregated' indicates that tensor is a result of
                        aggregation.
                        These tensors are sent to collaborators for the next
                        round.
                    - 'delta' indicates that value is a difference between
                        rounds for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            np.ndarray: aggregated tensor
        """
        for item in db_iterator:
            if "tags" in item and item["tags"] == ("secagg", ):
                print("saved tensor", item)
            if "tags" in item and item["tags"] == ("secagg", ) and item["tensor_name"] == "masks_sum":
                masks = item["nparray"]

        tensor_sum_with_masks = np.sum([tensor.tensor for tensor in local_tensors], axis=0)
        private_mask = masks[0]
        shared_mask = masks[1]

        return np.subtract(np.subtract(tensor_sum_with_masks, private_mask), shared_mask)
