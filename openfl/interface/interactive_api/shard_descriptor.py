# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shard descriptor."""

from typing import Iterable, List

import numpy as np


class ShardDataset:
    """Shard dataset class."""

    def __len__(self) -> int:
        """Return the len of the shard dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int):
        """Return an item by the index."""
        raise NotImplementedError


class ShardDescriptor:
    """Shard descriptor class."""

    def get_dataset(self, dataset_type: str) -> ShardDataset:
        """Return a shard dataset by type."""
        raise NotImplementedError

    @property
    def sample_shape(self) -> List[int]:
        """Return the sample shape info."""
        raise NotImplementedError

    @property
    def target_shape(self) -> List[int]:
        """Return the target shape info."""
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return ""


class DummyShardDataset(ShardDataset):
    """Dummy shard dataset class."""

    def __init__(self, *, size: int, sample_shape: List[int], target_shape: List[int]):
        """Initialize DummyShardDataset."""
        self.size = size
        self.samples = np.random.randint(0, 255, (self.size, *sample_shape), np.uint8)
        self.targets = np.random.randint(0, 255, (self.size, *target_shape), np.uint8)

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return self.size

    def __getitem__(self, index: int):
        """Return a item by the index."""
        return self.samples[index], self.targets[index]


class DummyShardDescriptor(ShardDescriptor):
    """Dummy shard descriptor class."""

    def __init__(
        self,
        sample_shape: Iterable[str],
        target_shape: Iterable[str],
        size: int,
    ) -> None:
        """Initialize DummyShardDescriptor."""
        self._sample_shape = [int(dim) for dim in sample_shape]
        self._target_shape = [int(dim) for dim in target_shape]
        self.size = size

    def get_dataset(self, dataset_type: str) -> ShardDataset:
        """Return a shard dataset by type."""
        return DummyShardDataset(
            size=self.size,
            sample_shape=self._sample_shape,
            target_shape=self._target_shape,
        )

    @property
    def sample_shape(self) -> List[int]:
        """Return the sample shape info."""
        return self._sample_shape

    @property
    def target_shape(self) -> List[int]:
        """Return the target shape info."""
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return "Dummy shard descriptor"
