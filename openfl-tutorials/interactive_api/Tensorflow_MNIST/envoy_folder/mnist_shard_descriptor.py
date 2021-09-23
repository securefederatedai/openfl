# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import logging

import numpy as np
from tensorflow import keras

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class MnistShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        (x_train, y_train), (x_test, y_test) = self.download_data()
        self.x_train = x_train[self.rank - 1::self.worldsize]
        self.y_train = y_train[self.rank - 1::self.worldsize]
        self.x_test = x_test[self.rank - 1::self.worldsize]
        self.y_test = y_test[self.rank - 1::self.worldsize]
        self.dataset_mode = None

    def set_dataset_type(self, mode='train'):
        """Select training or testing mode."""
        self.dataset_mode = mode

    def get_train_size(self):
        """Return train dataset size."""
        return len(self.x_train)

    def get_test_size(self):
        """Return test dataset size."""
        return len(self.x_test)

    def download_data(self):
        """Download prepared dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
        return (x_train, y_train), (x_test, y_test)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        if self.dataset_mode == 'train':
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_test[index], self.y_test[index]

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['784']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Mnist dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def __len__(self):
        """Return the len of the dataset."""
        if self.dataset_mode is None:
            return 0

        if self.dataset_mode == 'train':
            return len(self.x_train)
        else:
            return len(self.x_test)
