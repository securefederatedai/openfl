# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

from openfl.federated import KerasDataLoader


class KerasHippmapp3rsynth(KerasDataLoader):
    """Data Loader for synthetic Hippmapp3r Dataset."""

    def __init__(self, data_path=None, batch_size=32, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset. If None, initialize for model creation only.
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)
        self.feature_shape = [160, 160, 128]  # Hippmapp3r shape for Keras (channels last)

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        X_train = glob(f"{data_path}/X*.npy")
        y_train = glob(f"{data_path}/y*.npy")
        # Perform test-train split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        self.X_train = np.asarray(X_train)
        self.X_valid = np.asarray(X_valid)
        self.y_train = np.asarray(y_train)
        self.y_valid = np.asarray(y_valid)

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Returns:
            list: The shape of an example feature array [64, 64, 128] for Hippmapp3r.
        """
        return [64, 64, 128]

    def get_num_classes(self):
        """Returns the number of classes for classification tasks.

        Returns:
            int: The number of classes (2 for binary segmentation).
        """
        return 2  # Binary segmentation task (background and hippocampus)

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
            x_list = []
            y_list = []
            for _x, _y in zip(X[idxs[a:b]], y[idxs[a:b]]):
                x_list.append(np.load(_x))
                y_list.append(np.load(_y))
            yield np.stack(x_list), np.stack(y_list)
