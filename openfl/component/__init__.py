# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.component package."""

from .assigner import Assigner, RandomGroupedAssigner, StaticGroupedAssigner
from .aggregator import Aggregator
from .collaborator import Collaborator

__all__ = [
    'Assigner',
    'RandomGroupedAssigner',
    'StaticGroupedAssigner',
    'Aggregator',
    'Collaborator'
]
