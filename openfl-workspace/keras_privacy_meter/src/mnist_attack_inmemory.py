# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import TensorFlowDataLoader

from .mnist_utils import load_mnist_shard


class MNISTAttackInMemory(TensorFlowDataLoader):
    """TensorFlow Data Loader for MNIST Dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and
        #  what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size of
        # collaborator list.

        input_shape, num_classes, X_train, y_train, X_valid, y_valid = load_mnist_shard(
            shard_num=int(data_path), **kwargs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes

        #TODO MNIST data must be serialized to .txt and txt.npy file before this will work
        dataset_path = 'mnist.txt'
        saved_path = 'mnist_train.txt.npy'

        self.attack_data_handler = ml_privacy_meter.utils.attack_data(
                                       dataset_path=dataset_path,
                                       member_dataset_path=saved_path,
                                       batch_size=100,
                                       attack_percentage=10, input_shape=input_shape,
                                       normalization=True)
