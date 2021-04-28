# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregation functions package."""

from .median import Median
from .weighted_average import WeightedAverage
from .geometric_median import GeometricMedian
from .interface import AggregationFunctionInterface

__all__ = ['Median', 'WeightedAverage', 'GeometricMedian', 'AggregationFunctionInterface']
