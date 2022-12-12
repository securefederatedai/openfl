# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Noisy-Sin Shard Descriptor."""

from typing import List

import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class RegressionShardDescriptor(ShardDescriptor):
    """Regression Shard descriptor class."""

    def __init__(self, rank_worldsize: str = '1, 1', **kwargs) -> None:
        """
        Initialize Regression Data Shard Descriptor.

        This Shard Descriptor generate random regression data with some gaussian centered noise
        using make_regression method from sklearn.datasets.
        Shards data across participants using rank and world size.
        """

        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        X_train, y_train, X_test, y_test = self.generate_data()
        self.data_by_type = {
            'train': jnp.concatenate((X_train, y_train[:, None]), axis=1),
            'val': jnp.concatenate((X_test, y_test[:, None]), axis=1)
        }

    def generate_data(self):
        """Generate regression dataset with predefined params."""
        x, y = make_regression(n_samples=1000, n_features=1, noise=14, random_state=24)
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=24)
        self.data = jnp.concatenate((x, y[:, None]), axis=1)
        return X_train, y_train, X_test, y_test

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Incorrect dataset type: {dataset_type}')

        if dataset_type in ['train', 'val']:
            return self.data_by_type[dataset_type][self.rank - 1::self.worldsize]
        else:
            raise ValueError

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
        return (f'Regression dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
