# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorchDataLoader module."""
from .loader import DataLoader


class GaNDLFDataLoaderWrapper(DataLoader):
    """Data Loader for the Generally Nuanced Deep Learning Framework (GaNDLF)."""

    def __init__(self, data_path, feature_shape):
        self.train_csv = data_path + '/train.csv'
        self.val_csv = data_path + '/valid.csv'
        self.train_dataloader = None
        self.val_dataloader = None
        self.feature_shape = feature_shape

    def set_dataloaders(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def get_feature_shape(self):
        """Get the shape of an example feature array.

        Returns:
            tuple: shape of an example feature array
        """
        return self.feature_shape

    def get_train_loader(self, batch_size=None, num_batches=None):
        """
        Get training data loader.

        Returns
        -------
        loader object
        """
        return self.train_dataloader

    def get_valid_loader(self, batch_size=None):
        """
        Get validation data loader.

        Returns:
            loader object
        """
        return self.val_dataloader

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return len(self.train_dataloader.dataset)

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        return len(self.val_dataloader.dataset)
