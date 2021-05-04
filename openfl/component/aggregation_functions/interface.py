# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Aggregation function interface module."""
from typing import Iterator, Tuple
import numpy as np
from abc import ABC, abstractmethod

import pandas as pd


class AggregationFunctionInterface(ABC):
    """Interface for specifying aggregation function."""

    @abstractmethod
    def call(self,
             tensors: np.ndarray,
             weights: np.ndarray,
             db_iterator: Iterator[pd.Series],
             tensor_name: str,
             fl_round: int,
             tags: Tuple[str]) -> np.ndarray:
        """Aggregate tensors.

        Args:
            tensors: array of `np.ndarray`s of tensors to aggregate.
            weights: array of floats representing data partition (sum up to 1)
            db_iterator: iterator over history of aggregated versions of this tensor
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
