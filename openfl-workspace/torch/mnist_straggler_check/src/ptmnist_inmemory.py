# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import PyTorchDataLoader
from .mnist_utils import load_mnist_shard


class PyTorchMNISTInMemory(PyTorchDataLoader):
    """PyTorch data loader for MNIST dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and
        #  what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size
        # of collaborator list.
        try:
            int(data_path)
        except:
            raise ValueError(
                "Expected `%s` to be representable as `int`, as it refers to the data shard " +
                "number used by the collaborator.",
                data_path
            )

        num_classes, X_train, y_train, X_valid, y_valid = load_mnist_shard(
            shard_num=int(data_path), **kwargs
        )
        self.X_train = X_train
        self.y_train = y_train
        self.train_loader = self.get_train_loader()

        self.X_valid = X_valid
        self.y_valid = y_valid
        self.val_loader = self.get_valid_loader()

        self.num_classes = num_classes
