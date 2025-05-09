# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import KerasDataLoader

from .mnist_utils import load_mnist_shard


class KerasMNISTInMemory(KerasDataLoader):
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

        # Set MNIST-specific default attributes
        self.feature_shape = [28, 28, 1]  # MNIST shape for Keras (channels last)
        self.num_classes = 10  # MNIST has 10 classes

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        # Load actual data if a data path is provided
        try:
            int(data_path)
        except ValueError:
            raise ValueError(
                f"Expected '{data_path}' to be representable as `int`, "
                "as it refers to the data shard number used by the collaborator."
            )

        # Pass the feature_shape and num_classes to load_mnist_shard
        X_train, y_train, X_valid, y_valid = load_mnist_shard(
            shard_num=int(data_path),
            feature_shape=self.feature_shape,
            num_classes=self.num_classes,
            **kwargs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
