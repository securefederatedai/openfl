# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Datasets."""

import tensorflow as tf

from openfl.interface.interactive_api.experiment import DataInterface


class FedDataset(DataInterface):
    """Federation dataset."""

    def __init__(self, X_train, y_train, X_valid, y_valid, **kwargs):
        """Initialize Federation dataset."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs
        self._setup_datasets()

    def _setup_datasets(self):
        """Set datasets."""
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.X_valid, self.y_valid))
        self.valid_dataset = self.valid_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

    def _delayed_init(self, data_path='1,1'):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        data_path variable will be set according to data.yaml.
        """
        # With the next command the local dataset will be loaded on the collaborator node
        # For this example we have the same dataset on the same path, and we will shard it
        # So we use `data_path` information for this purpose.
        self.rank, self.world_size = [int(part) for part in data_path.split(',')]

        # Do the actual sharding
        self._do_sharding(self.rank, self.world_size)

    def _do_sharding(self, rank, world_size):
        """Do sharding."""
        self.X_train = self.X_train[rank - 1:: world_size]
        self.y_train = self.y_train[rank - 1:: world_size]
        self.X_valid = self.X_valid[rank - 1:: world_size]
        self.y_valid = self.y_valid[rank - 1:: world_size]
        self._setup_datasets()

    def get_train_loader(self, **kwargs):
        """Output of this method will be provided to tasks with optimizer in contract."""
        return self.train_dataset

    def get_valid_loader(self, **kwargs):
        """Output of this method will be provided to tasks without optimizer in contract."""
        return self.valid_dataset

    def get_train_data_size(self):
        """Information for aggregation."""
        return len(self.X_train)

    def get_valid_data_size(self):
        """Information for aggregation."""
        return len(self.X_valid)
