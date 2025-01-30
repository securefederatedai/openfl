# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""KerasDataLoader module."""

import numpy as np

from openfl.federated.data.loader import DataLoader


class KerasDataLoader(DataLoader):
    """A class used to represent a Federation Data Loader for Keras models.

    Attributes:
        batch_size (int): Size of batches used for all data loaders.
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_valid (np.array): Validation features.
        y_valid (np.array): Validation labels.
    """

    def __init__(self, batch_size, **kwargs):
        """Initializes the KerasDataLoader object.

        Args:
            batch_size (int): The size of batches used for all data loaders.
            kwargs: Additional arguments to pass to the function.
        """
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

        # Child classes should have init signature:
        # (self, batch_size, **kwargs), should call this __init__ and then
        # define self.X_train, self.y_train, self.X_valid, and self.y_valid

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Returns:
            tuple: The shape of an example feature array.
        """
        return self.X_train[0].shape

    def get_train_loader(self, batch_size=None, num_batches=None):
        """Returns the data loader for the training data.

        Args:
            batch_size (int, optional): The batch size for the data loader
                (default is None).
            num_batches (int, optional): The number of batches for the data
                loader (default is None).

        Returns:
            DataLoader: The DataLoader object for the training data.
        """
        return self._get_batch_generator(
            X=self.X_train,
            y=self.y_train,
            batch_size=batch_size,
            num_batches=num_batches,
        )

    def get_valid_loader(self, batch_size=None):
        """Returns the data loader for the validation data.

        Args:
            batch_size (int, optional): The batch size for the data loader
                (default is None).

        Returns:
            DataLoader: The DataLoader object for the validation data.
        """
        return self._get_batch_generator(X=self.X_valid, y=self.y_valid, batch_size=batch_size)

    def get_train_data_size(self):
        """Returns the total number of training samples.

        Returns:
            int: The total number of training samples.
        """
        return self.X_train.shape[0]

    def get_valid_data_size(self):
        """Returns the total number of validation samples.

        Returns:
            int: The total number of validation samples.
        """
        return self.X_valid.shape[0]

    @staticmethod
    def _batch_generator(X, y, idxs, batch_size, num_batches):
        """Generates batches of data.

        Args:
            X (np.array): The input data.
            y (np.array): The label data.
            idxs (np.array): The index of the dataset.
            batch_size (int): The batch size for the data loader.
            num_batches (int): The number of batches.

        Yields:
            tuple: The input data and label data for each batch.
        """
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            yield X[idxs[a:b]], y[idxs[a:b]]

    def _get_batch_generator(self, X, y, batch_size, num_batches=None):
        """Returns the dataset generator.

        Args:
            X (np.array): The input data.
            y (np.array): The label data.
            batch_size (int): The batch size for the data loader.
            num_batches (int, optional): The number of batches (default is
                None).

        Returns:
            generator: The dataset generator.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # shuffle data indices
        idxs = np.random.permutation(np.arange(X.shape[0]))

        if num_batches is None:
            # compute the number of batches
            num_batches = int(np.ceil(X.shape[0] / batch_size))

        # build the generator and return it
        return self._batch_generator(X, y, idxs, batch_size, num_batches)
