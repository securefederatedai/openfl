# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.component package."""

from .aggregator import Aggregator
from .assigner import Assigner
from .assigner import RandomGroupedAssigner
from .assigner import StaticGroupedAssigner
from .collaborator import Collaborator

__all__ = [
    'Assigner',
    'RandomGroupedAssigner',
    'StaticGroupedAssigner',
    'Aggregator',
    'Collaborator'
]
