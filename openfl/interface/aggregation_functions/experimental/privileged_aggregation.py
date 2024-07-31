# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Aggregation function interface module."""
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd

from openfl.interface.aggregation_functions import AggregationFunction
from openfl.utilities import LocalTensor


class PrivilegedAggregationFunction(AggregationFunction):
    """Privileged Aggregation Function interface provides write access to TensorDB Dataframe."""

    def __init__(self) -> None:
        """Initialize with TensorDB write access"""
        super().__init__()
        self._privileged = True

    @abstractmethod
    def call(
        self,
        local_tensors: List[LocalTensor],
        tensor_db: pd.DataFrame,
        tensor_name: str,
        fl_round: int,
        tags: Tuple[str],
    ) -> np.ndarray:
        """Aggregate tensors.

        Args:
            local_tensors (list[openfl.utilities.LocalTensor]): List of local
                tensors to aggregate.
            tensor_db: Raw TensorDB dataframe (for write access). Columns:
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
        raise NotImplementedError
