# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import TensorFlowDataLoader

from .tfds_utils import load_histology_shard


class TensorFlowHistologyInMemory(TensorFlowDataLoader):
    """TensorFlow Data Loader for Colorectal Histology Dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        _, num_classes, X_train, y_train, X_valid, y_valid = load_histology_shard(
            shard_num=data_path,
            categorical=False, **kwargs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes
