# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""FederatedDataset module."""

import numpy as np

from openfl.federated.data.loader_pt import PyTorchDataLoader
from openfl.utilities.data_splitters import EqualNumPyDataSplitter, NumPyDataSplitter


class FederatedDataSet(PyTorchDataLoader):
    """A Data Loader class used to represent a federated dataset for in-memory
    Numpy data.

    Attributes:
        train_splitter (NumPyDataSplitter): An object that splits the training
            data.
        valid_splitter (NumPyDataSplitter): An object that splits the
            validation data.
    """

    train_splitter: NumPyDataSplitter
    valid_splitter: NumPyDataSplitter

    def __init__(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        batch_size=1,
        num_classes=None,
        train_splitter=None,
        valid_splitter=None,
    ):
        """
        Initializes the FederatedDataSet object.

        Args:
            X_train (np.array): The training features.
            y_train (np.array): The training labels.
            X_valid (np.array): The validation features.
            y_valid (np.array): The validation labels.
            batch_size (int, optional): The batch size for the data loader.
                Defaults to 1.
            num_classes (int, optional): The number of classes the model will
                be trained on. Defaults to None.
            train_splitter (NumPyDataSplitter, optional): The object that
                splits the training data. Defaults to None.
            valid_splitter (NumPyDataSplitter, optional): The object that
                splits the validation data. Defaults to None.
        """
        super().__init__(batch_size)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        if num_classes is None:
            num_classes = np.unique(self.y_train).shape[0]
            print(f"Inferred {num_classes} classes from the provided labels...")
        self.num_classes = num_classes
        self.train_splitter = self._get_splitter_or_default(train_splitter)
        self.valid_splitter = self._get_splitter_or_default(valid_splitter)

    @staticmethod
    def _get_splitter_or_default(value):
        """Returns the provided splitter if it's a NumPyDataSplitter, otherwise
        returns a default EqualNumPyDataSplitter.

        Args:
            value (NumPyDataSplitter): The provided data splitter.

        Raises:
            NotImplementedError: If the provided data splitter is not a
                NumPyDataSplitter.
        """
        if value is None:
            return EqualNumPyDataSplitter()
        if isinstance(value, NumPyDataSplitter):
            return value
        else:
            raise NotImplementedError(f"Data splitter {value} is not supported")

    def split(self, num_collaborators):
        """Splits the dataset into equal parts for each collaborator and
        returns a list of FederatedDataSet objects.

        Args:
            num_collaborators (int): The number of collaborators to split the
                dataset between.

        Returns:
            FederatedDataSets (list): A list of FederatedDataSet objects, each
                representing a slice of the dataset for a collaborator.
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
                valid_splitter=self.valid_splitter,
            )
            for i in range(num_collaborators)
        ]
