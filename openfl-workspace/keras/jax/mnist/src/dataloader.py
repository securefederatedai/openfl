# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import KerasDataLoader
from .mnist_utils import load_mnist_shard


class JAXMNISTInMemory(KerasDataLoader):
    """Data Loader for MNIST Dataset."""

    def __init__(self, data_path=None, batch_size=32, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset. If None, initialize for model creation only.
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # Set default values for model initialization
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        try:
            int(data_path)
        except ValueError:
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

    def get_num_classes(self):
        """
        Return the number of classes for the dataset.
        Returns:
            int: Number of classes for the dataset
        """
        return 10

    def get_feature_shape(self):
        """
        Return the input shape for the model.
        Returns:
            list: The input shape for the model [28, 28, 1]
        """
        return [28, 28, 1]
