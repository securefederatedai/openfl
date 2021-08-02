# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FederatedDataset module."""

import numpy as np

from openfl.plugins.data_splitters import EqualNumPyDataSplitter
from openfl.plugins.data_splitters import EqualPyTorchDatasetSplitter
from openfl.plugins.data_splitters import NumPyDataSplitter
from openfl.plugins.data_splitters import PyTorchDatasetSplitter
from . import DataLoader
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

    data_splitter: NumPyDataSplitter

    def __init__(self, X_train, y_train, X_valid, y_valid,
                 batch_size=1, num_classes=None, data_splitter=None):
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
            print(f'Inferred {num_classes} classes from the provided labels...')
        self.num_classes = num_classes
        if data_splitter is None:
            self.data_splitter = EqualNumPyDataSplitter()
        elif isinstance(data_splitter, NumPyDataSplitter):
            self.data_splitter = data_splitter
        else:
            raise NotImplementedError(f'Data splitter {data_splitter} is not supported')

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
        train_idx = self.data_splitter.split(self.y_train, num_collaborators)
        valid_idx = self.data_splitter.split(self.y_valid, num_collaborators)

        return [
            FederatedDataSet(
                self.X_train[train_idx[i]],
                self.y_train[train_idx[i]],
                self.X_valid[valid_idx[i]],
                self.y_valid[valid_idx[i]],
                batch_size=self.batch_size,
                num_classes=self.num_classes
            ) for i in range(num_collaborators)
        ]


class PyTorchFederatedDataset(DataLoader):
    """FederatedDataset for PyTorch Datasets."""

    data_splitter: PyTorchDatasetSplitter

    def __init__(self, train_set, valid_set, batch_size, data_splitter=None):
        """Initialize.

        Args:
            train_set(torch.utils.data.Dataset): Training set.
            valid_set(torch.utils.data.Dataset): Validation set.
        """
        self.train_set = train_set
        self.valid_set = valid_set
        self.batch_size = batch_size
        if data_splitter is None:
            self.data_splitter = EqualPyTorchDatasetSplitter
        elif isinstance(data_splitter, PyTorchDatasetSplitter):
            self.data_splitter = data_splitter
        else:
            raise ValueError('data_splitter should either be None'
                             + ' or inherit from openfl.plugins.data_splitters.PyTorchDatasetSplitter')

    def get_feature_shape(self):
        """
        Get the shape of an example feature array.

        Returns:
            tuple: shape of an example feature array
        """
        return self.train_set[0][0].shape

    def get_train_loader(self, batch_size=None, num_batches=None):
        """
        Get training data loader.

        Returns:
            loader object (class defined by inheritor)
        """
        import torch.utils.data
        return torch.utils.data.DataLoader(self.train_set, batch_size or self.batch_size)

    def get_valid_loader(self, batch_size=None):
        """
        Get validation data loader.

        Returns:
            loader object (class defined by inheritor)
        """
        import torch.utils.data
        return torch.utils.data.DataLoader(self.valid_set, batch_size or self.batch_size)

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return len(self.train_set)

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        return len(self.valid_set)

    def split(self, num_collaborators):
        """Create a Federated Dataset for each of the collaborators.

        Args:
            num_collaborators: int
                Collaborators to split the dataset between
        Returns:
            list[PyTorchFederatedDataset]: A dataset slice for each collaborator
        """
        train_split = self.data_splitter.split(self.train_set, num_collaborators)
        valid_split = self.data_splitter.split(self.valid_set, num_collaborators)

        return [
            PyTorchFederatedDataset(
                train_split[i],
                valid_split[i],
                batch_size=self.batch_size,
                data_splitter=self.data_splitter
            ) for i in range(num_collaborators)
        ]
