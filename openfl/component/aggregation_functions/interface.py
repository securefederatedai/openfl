# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Aggregation function interface module."""
from typing import Iterator, Tuple, Dict
import numpy as np
from abc import ABC, abstractmethod

import pandas as pd


class AggregationFunctionInterface(ABC):
    """Interface for specifying aggregation function."""

    @abstractmethod
    def call(self,
             agg_tensor_dict: Dict[str, np.ndarray],
             weights: np.ndarray,
             db_iterator: Iterator[pd.Series],
             tensor_name: str,
             fl_round: int,
             tags: Tuple[str]) -> np.ndarray:
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
        raise NotImplementedError

    def __call__(self, tensors,
                 weights,
                 db_iterator,
                 tensor_name,
                 fl_round,
                 tags):
        """Use magic function for ease."""
        return self.call(tensors, weights, db_iterator, tensor_name, fl_round, tags)
