# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from logging import getLogger

import numpy as np
from torchvision import datasets, transforms

from openfl.federated import PyTorchDataLoader

logger = getLogger(__name__)


class PyTorchMNISTInMemory(PyTorchDataLoader):
    """PyTorch data loader for MNIST dataset."""

    def __init__(self, data_path=None, batch_size=32, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data. If None, initialize for model creation only.
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # Set default values for model initialization
        self.train_loader = None
        self.val_loader = None
        self.feature_shape = [1, 28, 28]
        self.num_classes = 10

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            logger.info("Initializing dataloader for model creation only (no data loading)")
            return

        try:
            int(data_path)
        except ValueError:
            raise ValueError(
                f"Expected '{data_path}' to be representable as `int`, "
                "as it refers to the data shard number used by the collaborator."
            )

        X_train, y_train, X_valid, y_valid = load_mnist_shard(
            shard_num=int(data_path),
            feature_shape=self.feature_shape,
            num_classes=self.num_classes,
            **kwargs
        )
        self.X_train = X_train
        self.y_train = y_train
        self.train_loader = self.get_train_loader()

        self.X_valid = X_valid
        self.y_valid = y_valid
        self.val_loader = self.get_valid_loader()

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Returns:
            list: The shape of an example feature array [1, 28, 28] for MNIST.
        """
        return self.feature_shape

    def get_num_classes(self):
        """Returns the number of classes for classification tasks.

        Returns:
            int: The number of classes (10 for MNIST).
        """
        return self.num_classes


def load_mnist_shard(
    shard_num, collaborator_count, feature_shape=None, num_classes=None,
    categorical=False, channels_last=True, **kwargs
):
    """
    Load the MNIST dataset.

    Args:
        shard_num (int): The shard to use from the dataset
        collaborator_count (int): The number of collaborators in the
                                  federation
        feature_shape (list, optional): The shape of input features.
        num_classes (int, optional): Number of classes.
        categorical (bool): True = convert the labels to one-hot encoded
                            vectors (Default = True)
        channels_last (bool): True = The input images have the channels
                              last (Default = True)
        **kwargs: Additional parameters to pass to the function

    Returns:
        numpy.ndarray: The training data
        numpy.ndarray: The training labels
        numpy.ndarray: The validation data
        numpy.ndarray: The validation labels
    """


    (X_train, y_train), (X_valid, y_valid) = _load_raw_datashards(
        shard_num, collaborator_count, transform=transforms.ToTensor()
    )

    logger.info(f"MNIST > X_train Shape : {X_train.shape}")
    logger.info(f"MNIST > y_train Shape : {y_train.shape}")
    logger.info(f"MNIST > Train Samples : {X_train.shape[0]}")
    logger.info(f"MNIST > Valid Samples : {X_valid.shape[0]}")

    if categorical:
        # convert class vectors to binary class matrices
        y_train = one_hot(y_train, num_classes)
        y_valid = one_hot(y_valid, num_classes)

    return X_train, y_train, X_valid, y_valid


def one_hot(labels, classes):
    """
    One Hot encode a vector.

    Args:
        labels (list):  List of labels to onehot encode
        classes (int): Total number of categorical classes

    Returns:
        np.array: Matrix of one-hot encoded labels
    """
    return np.eye(classes)[labels]


def _load_raw_datashards(shard_num, collaborator_count, transform=None):
    """
    Load the raw data by shard.

    Returns tuples of the dataset shard divided into training and validation.

    Args:
        shard_num (int): The shard number to use
        collaborator_count (int): The number of collaborators in the federation
        transform: torchvision.transforms.Transform to apply to images

    Returns:
        2 tuples: (image, label) of the training, validation dataset
    """
    train_data, val_data = (
        datasets.MNIST("data", train=train, download=True, transform=transform)
        for train in (True, False)
    )
    X_train_tot, y_train_tot = train_data.train_data, train_data.train_labels
    X_valid_tot, y_valid_tot = val_data.test_data, val_data.test_labels

    # create the shards
    shard_num = int(shard_num)
    X_train = X_train_tot[shard_num::collaborator_count].unsqueeze(1).float()
    y_train = y_train_tot[shard_num::collaborator_count]

    X_valid = X_valid_tot[shard_num::collaborator_count].unsqueeze(1).float()
    y_valid = y_valid_tot[shard_num::collaborator_count]

    return (X_train, y_train), (X_valid, y_valid)
