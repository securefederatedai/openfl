# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Straggler Handling functions package."""

from .cutoff_time_based_straggler_handling import (
    CutoffTimeBasedStragglerHandling,
)
from .percentage_based_straggler_handling import (
    PercentageBasedStragglerHandling,
)
from .straggler_handling_function import StragglerHandlingFunction

__all__ = [
    'CutoffTimeBasedStragglerHandling', 'PercentageBasedStragglerHandling',
    'StragglerHandlingFunction'
]
