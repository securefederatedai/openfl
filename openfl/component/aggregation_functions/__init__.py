# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregation functions package."""

from .geometric_median import GeometricMedian
from .interface import AggregationFunctionInterface
from .median import Median
from .weighted_average import WeightedAverage

__all__ = ['Median', 'WeightedAverage', 'GeometricMedian', 'AggregationFunctionInterface']
