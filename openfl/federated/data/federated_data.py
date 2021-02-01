# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FederatedDataset module."""

import numpy as np

from .loader_pt import PyTorchDataLoader


class FederatedDataSet(PyTorchDataLoader):
    """
    Data Loader for in memory Numpy data.

    Args:
        X_train: np.array
            Training Features
        y_train: np.array
            Training labels
        X_val: np.array
            Validation features
        y_val: np.array
            Validation labels
        batch_size : int
            The batch size for the data loader
        num_classes : int
            The number of classes the model will be trained on
        **kwargs: Additional arguments to pass to the function

    """

    def __init__(self, X_train, y_train, X_valid, y_valid,
                 batch_size=1, num_classes=None, **kwargs):
        """
        Initialize.

        Args:
            X_train: np.array
                Training Features
            y_train: np.array
                Training labels
            X_val: np.array
                Validation features
            y_val: np.array
                Validation labels
            batch_size : int
                The batch size for the data loader
            num_classes : int
                The number of classes the model will be trained on
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(batch_size)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        if num_classes is None:
            num_classes = np.unique(self.y_train).shape[0]
            print('Inferred {} classes from the provided'
                  ' labels...'.format(num_classes))
        self.num_classes = num_classes

    def split(self, num_collaborators, shuffle=True, equally=False):
        """Create a Federated Dataset for each of the collaborators.

        Args:
            num_collaborators: int
                Collaborators to split the dataset between
            shuffle: boolean
                Should the dataset be randomized?
            equally: boolean
                Should each collaborator get the same amount of data?

        Returns:
            list[FederatedDataSets]
                A dataset slice for each collaborator
        """
        # collaborator_datasets = []

        if shuffle:
            train_shuffle = np.random.choice(
                len(self.X_train), len(self.X_train), replace=False
            )
            self.X_train = self.X_train[train_shuffle]
            self.y_train = self.y_train[train_shuffle]
            val_shuffle = np.random.choice(
                len(self.X_valid), len(self.X_valid), replace=False
            )
            self.X_valid = self.X_valid[val_shuffle]
            self.y_valid = self.y_valid[val_shuffle]

        # train_idx = 0
        # val_idx = 0

        if equally:
            X_train = np.array_split(self.X_train, num_collaborators)
            y_train = np.array_split(self.y_train, num_collaborators)
            X_valid = np.array_split(self.X_valid, num_collaborators)
            y_valid = np.array_split(self.y_valid, num_collaborators)
        else:
            train_split = np.sort(np.random.choice(
                len(self.X_train), num_collaborators - 1, replace=False)
            )
            val_split = np.sort(np.random.choice(
                len(self.X_val), num_collaborators - 1, replace=False)
            )
            X_train = np.split(self.X_train, train_split)
            y_train = np.split(self.y_train, train_split)
            X_valid = np.split(self.X_valid, val_split)
            y_valid = np.split(self.y_valid, val_split)

        return [
            FederatedDataSet(
                X_train[i],
                y_train[i],
                X_valid[i],
                y_valid[i],
                batch_size=self.batch_size,
                num_classes=self.num_classes
            ) for i in range(num_collaborators)
        ]
