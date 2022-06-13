# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Noisy-Sin Shard Descriptor."""

from typing import List

import jax.numpy as jnp

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class LinRegSD(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, rank: int, n_samples: int = 100, n_features : int = 1, noise : int = 5) -> None:
        """
        Initialize LinReg Shard Descriptor.

        This Shard Descriptor generate random regression data with some gaussian centered noise
        using make_regression method from sklearn.datasets.
        """
        self.metadata = {
            'rank' : rank,
            'n_features' : n_features,
            'n_samples' : n_samples,
            'noise' : noise
        }
        x, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=rank)
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=rank)
        self.data = jnp.concatenate((x, y[:, None]), axis=1)
        self.train = jnp.concatenate((X_train, y_train[:, None]), axis=1)
        self.test = jnp.concatenate((X_test, y_test[:, None]), axis=1)

    def get_dataset(self, dataset_type: str) -> jnp.ndarray:
        """
        Return a shard dataset by type.

        A simple list with elements (x, y) implements the Shard Dataset interface.
        """
        if dataset_type == 'train':
            return self.train
        elif dataset_type == 'val':
            return self.test
        else:
            return None

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        (*x, _) = self.data[0]
        return [str(i) for i in jnp.array(x, ndmin=1).shape]

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        (*_, y) = self.data[0]
        return [str(i) for i in jnp.array(y, ndmin=1).shape]

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return 'Allowed dataset types are `train` and `val`'
