# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Kvasir shard descriptor."""

import numpy as np
from tensorflow import keras

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class MNISTShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, rank_worldsize: str = '1,1') -> None:
        """Initialize KvasirShardDescriptor."""
        super().__init__()

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        # Sharding
        self.X_train = x_train[self.rank - 1::self.worldsize]
        self.y_train = y_train[self.rank - 1::self.worldsize]
        self.X_test = x_test[self.rank - 1::self.worldsize]
        self.y_test = y_test[self.rank - 1::self.worldsize]

        # Calculating data and target shapes
        sample, _ = self[0]
        self._sample_shape = [str(dim) for dim in sample.shape]
        self._target_shape = ['0']

    def __getitem__(self, index):
        """Return a item by the index."""
        if index < len(self.X_train):
            return self.X_train[index], self.y_train[index]
        index -= len(self.X_train) + 1
        return self.X_test[index], self.y_test[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.X_train) + len(self.X_test)

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
        return (f'MNIST dataset, shard number {self.rank}'
                f' out of {self.worldsize}')


if __name__ == '__main__':
    from openfl.interface.cli import setup_logging
    setup_logging()

    data_folder = 'data'
    rank_worldsize = '1,100'

    mnist_sd = MNISTShardDescriptor(
        rank_worldsize=rank_worldsize)

    print(mnist_sd.dataset_description)
    print(mnist_sd.sample_shape, mnist_sd.target_shape)

    from openfl.component.envoy.envoy import Envoy

    shard_name = 'one'
    director_uri = 'localhost:50051'

    keeper = Envoy(
        shard_name=shard_name,
        director_uri=director_uri,
        shard_descriptor=mnist_sd,
        tls=False,
        root_ca='./cert/root_ca.crt',
        key='./cert/one.key',
        cert='./cert/one.crt',
    )

    keeper.start()
