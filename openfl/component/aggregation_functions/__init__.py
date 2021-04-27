# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregation functions package."""

from .median import median
from .fedavg import weighted_average
from .geometric_median import geometric_median

__all__ = ['median', 'weighted_average', 'geometric_median']
