# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shard descriptor."""


import numpy as np


class ShardDescriptor:
    """Shard descriptor class."""

    def __len__(self):
        """Return the len of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int):
        """Return a item by the index."""
        raise NotImplementedError

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        raise NotImplementedError

    @property
    def target_shape(self):
        """Return the target shape info."""
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return ''


class DummyShardDescriptor(ShardDescriptor):
    """Dummy shard descriptor class."""

    def __init__(self, sample_shape, target_shape, size) -> None:
        """Initialize DummyShardDescriptor."""
        self._sample_shape = [int(dim) for dim in sample_shape]
        self._target_shape = [int(dim) for dim in target_shape]
        self.size = size
        self.samples = np.random.randint(0, 255, (self.size, *self.sample_shape), np.uint8)
        self.targets = np.random.randint(0, 255, (self.size, *self.target_shape), np.uint8)

    def __len__(self):
        """Return the len of the dataset."""
        return self.size

    def __getitem__(self, index: int):
        """Return a item by the index."""
        return self.samples[index], self.targets[index]

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return self._sample_shape

    @property
    def target_shape(self):
        """Return the target shape info."""
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return 'Dummy shard descriptor'
