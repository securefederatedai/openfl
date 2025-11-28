# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Adaptive aggregation module."""

from typing import List

import numpy as np

from openfl.interface.aggregation_functions.core.interface import AggregationFunction
from openfl.utilities.optimizers.numpy.base_optimizer import Optimizer
from openfl.utilities.types import LocalTensor


class AdaptiveAggregation(AggregationFunction):
    """Adaptive Federated Aggregation funtcion.

    According to https://arxiv.org/abs/2003.00295
    """

    def __init__(
        self,
        optimizer: Optimizer,
        agg_func: AggregationFunction,
    ) -> None:
        """Initialize the AdaptiveAggregation class.

        Args:
            optimizer (Optimizer): One of numpy optimizer class instance.
            agg_func (AggregationFunction): Aggregate function for aggregating
                parameters that are not inside the optimizer.
        """
        super().__init__()
        self.optimizer = optimizer
        self.default_agg_func = agg_func

    @staticmethod
    def _make_gradient(
        base_model_nparray: np.ndarray, local_tensors: List[LocalTensor]
    ) -> np.ndarray:
        """Make gradient.

        Args:
            base_model_nparray (np.ndarray): The base model tensor.
            local_tensors (List[LocalTensor]): List of local tensors.

        Returns:
            np.ndarray: The gradient tensor.
        """
        return sum(
            [
                local_tensor.weight * (base_model_nparray - local_tensor.tensor)
                for local_tensor in local_tensors
            ]
        )

    def call(self, local_tensors, db_iterator, tensor_name, fl_round, tags) -> np.ndarray:
        """Aggregate tensors.

        Args:
            local_tensors (list[openfl.utilities.LocalTensor]): List of local
                tensors to aggregate.
            db_iterator: An iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight','fc2.bias'.
                - 'fl_round': 0-based number of round corresponding to this
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
        if tensor_name not in self.optimizer.params:
            return self.default_agg_func(local_tensors, db_iterator, tensor_name, fl_round, tags)

        base_model_nparray = None
        search_tag = "aggregated" if fl_round != 0 else "model"
        for record in db_iterator:
            if (
                record["round"] == fl_round
                and record["tensor_name"] == tensor_name
                and search_tag in record["tags"]
                and "delta" not in record["tags"]
            ):
                base_model_nparray = record["nparray"]

        if base_model_nparray is None:
            raise KeyError(
                f"There is no current global model in TensorDB for tensor name: {tensor_name}"
            )

        gradient = self._make_gradient(base_model_nparray, local_tensors)
        gradients = {tensor_name: gradient}
        self.optimizer.step(gradients)
        return self.optimizer.params[tensor_name]
