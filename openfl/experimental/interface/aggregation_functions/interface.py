# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Aggregation function interface module."""
from abc import abstractmethod

from openfl.utilities import SingletonABCMeta

class AggregationFunction(metaclass=SingletonABCMeta):
    """Abstract base class for specifying aggregation functions."""

    @abstractmethod
    def aggregate_models(self, **kwargs):
        """Aggregate training results using algorithms"""
        raise NotImplementedError

    @abstractmethod
    def aggregate_metrics(self, **kwargs):
        """Aggregate evaluation metrics"""
        raise NotImplementedError
