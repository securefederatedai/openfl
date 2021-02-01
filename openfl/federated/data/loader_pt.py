# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorchDataLoader module."""

import numpy as np

from .loader import DataLoader
from math import ceil


class PyTorchDataLoader(DataLoader):
    """Federation Data Loader for TensorFlow Models."""

    def __init__(self, batch_size, random_seed=None, **kwargs):
        """
        Instantiate the data object.

        Args:
            batch_size: Size of batches used for all data loaders
            kwargs: consumes all un-used kwargs

        Returns:
            None
        """
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.random_seed = random_seed

        # Child classes should have init signature:
        # (self, batch_size, **kwargs), should call this __init__ and then
        # define self.X_train, self.y_train, self.X_valid, and self.y_valid

    def get_feature_shape(self):
        """Get the shape of an example feature array.

        Returns:
            tuple: shape of an example feature array
        """
        return self.X_train[0].shape

    def get_train_loader(self, batch_size=None, num_batches=None):
        """
        Get training data loader.

        Returns
        -------
        loader object
        """
        return self._get_batch_generator(
            X=self.X_train, y=self.y_train, batch_size=batch_size, num_batches=num_batches)

    def get_valid_loader(self, batch_size=None):
        """
        Get validation data loader.

        Returns:
            loader object
        """
        return self._get_batch_generator(X=self.X_valid, y=self.y_valid, batch_size=batch_size)

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return self.X_train.shape[0]

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        return self.X_valid.shape[0]

    @staticmethod
    def _batch_generator(X, y, idxs, batch_size, num_batches):
        """
        Generate batch of data.

        Args:
            X: input data
            y: label data
            idxs: The index of the dataset
            batch_size: The batch size for the data loader
            num_batches: The number of batches

        Yields:
            tuple: input data, label data

        """
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            yield X[idxs[a:b]], y[idxs[a:b]]

    def _get_batch_generator(self, X, y, batch_size, num_batches=None):
        """
        Return the dataset generator.

        Args:
            X: input data
            y: label data
            batch_size: The batch size for the data loader

        """
        if batch_size is None:
            batch_size = self.batch_size

        # shuffle data indices
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        idxs = np.random.permutation(np.arange(X.shape[0]))

        # compute the number of batches
        if num_batches is None:
            num_batches = ceil(X.shape[0] / batch_size)

        # build the generator and return it
        return self._batch_generator(X, y, idxs, batch_size, num_batches)
