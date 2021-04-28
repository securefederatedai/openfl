# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Median module."""

from .interface import AggregationFunctionInterface
import numpy as np


def median(tensors):
    """Compute median."""
    return np.median(tensors, axis=0)


class Median(AggregationFunctionInterface):
    def __call__(self, tensors: np.ndarray, **kwargs) -> np.ndarray:
        return median(tensors)
