# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import logging
from pathlib import Path
from typing import Tuple

from base import Dataset
from base import ShardDescriptor


import numpy as np
from numpy import ndarray


from tensorflow import keras


logger = logging.getLogger(__name__)


class MnistDataset(Dataset):
    """Mnist dataset class."""


    def __init__(self, data_folder: Path, X_data, y_labels, data_type='train'):
        """Initialize Mnist Dataset."""
        self._common_data_folder = data_folder
        self.X_data = X_data
        self.y_labels = y_labels
        self.data_type = data_type
        assert len(X_data) == len(y_labels), 'Count of data samples and labels must be the same'

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.y_labels)

    def __getitem__(self, index: int) -> Tuple[ndarray, int]:
        """Return an item by the index."""
        x = self.X_data[index]
        label = self.y_labels[index]
        return x, label


class MnistShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(
            self,
            data_folder: str = 'data',
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.data_folder = Path.cwd() / data_folder
        (x_train, y_train), (x_test, y_test) = self.download_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

    def download_data(self):
        """Download prepared dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
        return (x_train, y_train), (x_test, y_test)

    def get_dataset(self, dataset_type):
        """Return a dataset by type."""
        if dataset_type == 'train':
            self.rank, self.worldsize
            return MnistDataset(
                self.data_folder,
                self.x_train[self.rank - 1::self.worldsize, :],
                self.y_train[self.rank - 1::self.worldsize],
                data_type=dataset_type
            )
        return MnistDataset(
                self.data_folder,
                self.x_test[self.rank - 1::self.worldsize, :],
                self.y_test[self.rank - 1::self.worldsize],
                data_type=dataset_type,
            )

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
        return 0
