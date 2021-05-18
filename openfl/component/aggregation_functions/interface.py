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
            db_iterator: iterator over history of all tensors.
                Columns: ['tensor_name', 'round', 'tags', 'nparray']
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        """
        raise NotImplementedError

    def __call__(self, local_tensors,
                 db_iterator,
                 tensor_name,
                 fl_round,
                 tags):
        """Use magic function for ease."""
        return self.call(local_tensors, db_iterator, tensor_name, fl_round, tags)
