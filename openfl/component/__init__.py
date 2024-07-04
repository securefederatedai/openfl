# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl.component package."""

from .aggregator import Aggregator
from .assigner import Assigner, RandomGroupedAssigner, StaticGroupedAssigner
from .collaborator import Collaborator
from .straggler_handling_functions import (
    CutoffTimeBasedStragglerHandling,
    PercentageBasedStragglerHandling,
    StragglerHandlingFunction,
)

__all__ = [
    'Assigner', 'RandomGroupedAssigner', 'StaticGroupedAssigner', 'Aggregator',
    'Collaborator', 'StragglerHandlingFunction',
    'CutoffTimeBasedStragglerHandling', 'PercentageBasedStragglerHandling'
]
