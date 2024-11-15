# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Noisy-Sin Shard Descriptor."""

from typing import List

import numpy as np

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class LinRegSD(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, rank: int, n_samples: int = 10, noise: float = 0.15) -> None:
        """
        Initialize LinReg Shard Descriptor.

        This Shard Descriptor generate random data. Sample features are
        floats between pi/3 and 5*pi/3, and targets are calculated
        calculated as sin(feature) + normal_noise.
        """
        np.random.seed(rank)  # Setting seed for reproducibility
        self.n_samples = max(n_samples, 5)
        self.interval = 240
        self.x_start = 60
        x = np.random.rand(n_samples, 1) * self.interval + self.x_start
        x *= np.pi / 180
        y = np.sin(x) + np.random.normal(0, noise, size=(n_samples, 1))
        self.data = np.concatenate((x, y), axis=1)

    def get_dataset(self, dataset_type: str) -> np.ndarray:
        """
        Return a shard dataset by type.

        A simple list with elements (x, y) implemets the Shard Dataset interface.
        """
        if dataset_type == 'train':
            return self.data[:self.n_samples // 2]
        elif dataset_type == 'val':
            return self.data[self.n_samples // 2:]
        else:
            pass

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        (*x, _) = self.data[0]
        return [str(i) for i in np.array(x, ndmin=1).shape]

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        (*_, y) = self.data[0]
        return [str(i) for i in np.array(y, ndmin=1).shape]

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return 'Allowed dataset types are `train` and `val`'
