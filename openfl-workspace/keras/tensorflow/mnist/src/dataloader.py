# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import KerasDataLoader
from .mnist_utils import load_mnist_shard


class MNISTInMemory(KerasDataLoader):
    """Data Loader for MNIST Dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        try:
            int(data_path)
        except:
            raise ValueError(
                "Expected `%s` to be representable as `int`, as it refers to the data shard " +
                "number used by the collaborator.",
                data_path
            )

        _, num_classes, X_train, y_train, X_valid, y_valid = load_mnist_shard(
            shard_num=int(data_path), **kwargs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes
