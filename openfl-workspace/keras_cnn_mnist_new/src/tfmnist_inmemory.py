# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
from tensorflow.keras.utils import Sequence

from openfl.federated import TensorFlowDataLoader
from .mnist_utils import load_mnist_shard


class TensorFlowMNISTInMemory(TensorFlowDataLoader):
    """TensorFlow Data Loader for MNIST Dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and
        #  what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size of
        # collaborator list.

        _, num_classes, X_train, y_train, X_valid, y_valid = load_mnist_shard(
            shard_num=int(data_path), **kwargs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes

    def get_train_loader(self, batch_size=None):
        """
        Get training data loader.

        Returns
        -------
        loader object
        """
        return DataGenerator(self.X_train, self.y_train, batch_size=batch_size)

    def get_valid_loader(self, batch_size=None):
        """
        Get validation data loader.

        Returns:
            loader object
        """
        return DataGenerator(self.X_valid, self.y_valid, batch_size=batch_size)


class DataGenerator(Sequence):

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.X[batch], self.y[batch]
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
