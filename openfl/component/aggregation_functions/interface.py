# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Aggregation function interface module."""
from typing import Iterator, Tuple, List
from openfl.utilities import LocalTensor, SingletonABCMeta
import numpy as np
from abc import abstractmethod

import pandas as pd


class AggregationFunctionInterface(metaclass=SingletonABCMeta):
    """Interface for specifying aggregation function."""

    @abstractmethod
    def call(self,
             local_tensors: List[LocalTensor],
             weights: np.ndarray,
             db_iterator: Iterator[pd.Series],
             tensor_name: str,
             fl_round: int,
             tags: Tuple[str]) -> np.ndarray:
        """Aggregate tensors.

        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training result.
                        These tensors are passed to the aggregator node after local learning.
                    - 'aggregated' indicates that tensor is a result of aggregation.
                        These tensors are sent to collaborators for the next round.
                    - 'delta' indicates that value is a difference between rounds
                        for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            np.ndarray: aggregated tensor
        """
        raise NotImplementedError

    def __call__(self, local_tensors,
                 db_iterator,
                 tensor_name,
                 fl_round,
                 tags):
        """Use magic function for ease."""
        return self.call(local_tensors, db_iterator, tensor_name, fl_round, tags)
