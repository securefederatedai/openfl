# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FederatedDataset module."""

import numpy as np

from openfl.utilities.data_splitters import EqualNumPyDataSplitter
from openfl.utilities.data_splitters import NumPyDataSplitter
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

    train_splitter: NumPyDataSplitter
    valid_splitter: NumPyDataSplitter

    def __init__(self, X_train, y_train, X_valid, y_valid,
                 batch_size=1, num_classes=None, train_splitter=None, valid_splitter=None):
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
            train_splitter: NumPyDataSplitter
                Data splitter for train dataset.
            valid_splitter: NumPyDataSplitter
                Data splitter for validation dataset.
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(batch_size)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        if num_classes is None:
            num_classes = np.unique(self.y_train).shape[0]
            print(f'Inferred {num_classes} classes from the provided labels...')
        self.num_classes = num_classes
        self.train_splitter = self._get_splitter_or_default(train_splitter)
        self.valid_splitter = self._get_splitter_or_default(valid_splitter)

    @staticmethod
    def _get_splitter_or_default(value):
        if value is None:
            return EqualNumPyDataSplitter()
        if isinstance(value, NumPyDataSplitter):
            return value
        else:
            raise NotImplementedError(f'Data splitter {value} is not supported')

    def split(self, num_collaborators):
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
        train_idx = self.train_splitter.split(self.y_train, num_collaborators)
        valid_idx = self.valid_splitter.split(self.y_valid, num_collaborators)

        return [
            FederatedDataSet(
                self.X_train[train_idx[i]],
                self.y_train[train_idx[i]],
                self.X_valid[valid_idx[i]],
                self.y_valid[valid_idx[i]],
                batch_size=self.batch_size,
                num_classes=self.num_classes,
                train_splitter=self.train_splitter,
                valid_splitter=self.valid_splitter
            ) for i in range(num_collaborators)
        ]
