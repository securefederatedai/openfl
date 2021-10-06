# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base shard descriptor."""


class ShardDataset:
    """Shard dataset class."""

    def __len__(self):
        """Return the len of the shard dataset."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Return an item by the index."""
        raise NotImplementedError


class ShardDescriptor:
    """Shard descriptor class."""

    def get_dataset(self, dataset_type) -> ShardDataset:
        """Return a shard dataset by type."""
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
