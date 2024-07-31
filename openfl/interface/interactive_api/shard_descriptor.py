# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


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
        """Return a shard dataset by type.

        Args:
            dataset_type (str): The type of the dataset.

        Returns:
            ShardDataset: The shard dataset.
        """
        raise NotImplementedError

    @property
    def sample_shape(self) -> List[int]:
        """Return the sample shape info.

        Returns:
            List[int]: The sample shape.
        """
        raise NotImplementedError

    @property
    def target_shape(self) -> List[int]:
        """Return the target shape info.

        Returns:
            List[int]: The target shape.
        """
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """Return the dataset description.

        Returns:
            str: The dataset description.
        """
        return ""


class DummyShardDataset(ShardDataset):
    """Dummy shard dataset class."""

    def __init__(self, *, size: int, sample_shape: List[int], target_shape: List[int]):
        """
        Initialize DummyShardDataset.

        Args:
            size (int): The size of the dataset.
            sample_shape (List[int]): The shape of the samples.
            target_shape (List[int]): The shape of the targets.
        """
        self.size = size
        self.samples = np.random.randint(0, 255, (self.size, *sample_shape), np.uint8)
        self.targets = np.random.randint(0, 255, (self.size, *target_shape), np.uint8)

    def __len__(self) -> int:
        """Return the len of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.size

    def __getitem__(self, index: int):
        """Return a item by the index.

        Args:
            index (int): The index of the item.

        Returns:
            tuple: The sample and target at the given index.
        """
        return self.samples[index], self.targets[index]


class DummyShardDescriptor(ShardDescriptor):
    """Dummy shard descriptor class."""

    def __init__(
        self,
        sample_shape: Iterable[str],
        target_shape: Iterable[str],
        size: int,
    ) -> None:
        """
        Initialize DummyShardDescriptor.

        Args:
            sample_shape (Iterable[str]): The shape of the samples.
            target_shape (Iterable[str]): The shape of the targets.
            size (int): The size of the dataset.
        """
        self._sample_shape = [int(dim) for dim in sample_shape]
        self._target_shape = [int(dim) for dim in target_shape]
        self.size = size

    def get_dataset(self, dataset_type: str) -> ShardDataset:
        """
        Return a shard dataset by type.

        Args:
            dataset_type (str): The type of the dataset.

        Returns:
            ShardDataset: The shard dataset.
        """
        return DummyShardDataset(
            size=self.size,
            sample_shape=self._sample_shape,
            target_shape=self._target_shape,
        )

    @property
    def sample_shape(self) -> List[int]:
        """Return the sample shape info.

        Returns:
            List[int]: The sample shape.
        """
        return self._sample_shape

    @property
    def target_shape(self) -> List[int]:
        """Return the target shape info.

        Returns:
            List[int]: The target shape.
        """
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description.

        Returns:
            str: The dataset description.
        """
        return "Dummy shard descriptor"
