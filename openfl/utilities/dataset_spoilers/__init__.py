# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dataset spoilers package."""

from .shard_corruptor import corrupt_shard
from .dataset_spoil_methods import spoil_targets_random_choice
from .dataset_spoil_methods import spoil_targets_rotation


__all__ = [
    'corrupt_shard',
    'spoil_targets_random_choice',
    'spoil_targets_rotation',
]
