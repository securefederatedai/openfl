# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shard descriptor."""

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor as SDBase


class ShardDescriptor(SDBase):
    """Example shard descriptor class."""

    def __init__(self, data_path) -> None:
        """Initialize a shard descriptor object."""
        super().__init__()
        self.data_path = data_path
        self.dataset_length = 100

    def __len__(self):
        """Return the len of the dataset."""
        return self.dataset_length

    def __getitem__(self, index: int):
        """Return a item by the index."""
        return None

    @property
    def sample_shape(self) -> list:
        """Return the sample shape info."""
        return []

    @property
    def target_shape(self) -> list:
        """Return the target shape info."""
        return []

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return self.data_path
