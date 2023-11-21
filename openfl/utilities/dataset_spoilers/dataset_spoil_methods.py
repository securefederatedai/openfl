# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dataset spoilers module."""

import random
from openfl.interface.interactive_api.shard_descriptor import ShardDataset


def spoil_targets_rotation(shard_dataset: ShardDataset) -> ShardDataset:
    """Spoiler method that shifts targets index by 1."""
    class SpoiledDataset(ShardDataset):
        def __getitem__(self, index):
            sample, _ = shard_dataset[index]
            _, target = shard_dataset[(index + 1) % len(self)]
            return sample, target

        def __len__(self):
            return len(shard_dataset)

    return SpoiledDataset()


def spoil_targets_random_choice(shard_dataset: ShardDataset) -> ShardDataset:
    """Spoiler method that takes a random target from the Dataset."""
    class SpoiledDataset(ShardDataset):
        def __getitem__(self, index):
            sample, _ = shard_dataset[index]
            _, target = random.choice(shard_dataset)
            return sample, target

        def __len__(self):
            return len(shard_dataset)

    return SpoiledDataset()
