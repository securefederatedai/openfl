# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Histogram module."""

import numpy as np

from openfl.interface.aggregation_functions.core import AggregationFunction


class X(AggregationFunction):
    

    def call(self, local_tensors, *_) -> np.ndarray:

        if not local_tensors:
            raise ValueError("Histogram result is empty.")

        
        return None
