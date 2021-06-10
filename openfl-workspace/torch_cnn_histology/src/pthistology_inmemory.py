# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import PyTorchDataLoader
from .histology_utils import load_histology_shard


class PyTorchHistologyInMemory(PyTorchDataLoader):
    """PyTorch data loader for Histology dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super init
             and load_mnist_shard
        """
        super().__init__(batch_size, random_seed=0, **kwargs)

        _, num_classes, X_train, y_train, X_valid, y_valid = load_histology_shard(
            shard_num=int(data_path), **kwargs)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes
