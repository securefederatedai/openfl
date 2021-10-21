# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from logging import getLogger
from pathlib import Path

import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

logger = getLogger(__name__)


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


def load_data(path='mnist.npz', cache_dir='~/.openfl/data'):
    """Download MNIST dataset."""
    cache_dir = Path(cache_dir).expanduser().absolute()
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file(
        path,
        origin=origin_folder + 'mnist.npz',
        file_hash='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1',
        cache_subdir=cache_dir)
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


def _load_raw_datashards(data_dir, shard_num, collaborator_count):
    """
    Load the raw data by shard.

    Returns tuples of the dataset shard divided into training and validation.

    Args:
        shard_num (int): The shard number to use
        collaborator_count (int): The number of collaborators in the federation

    Returns:
        2 tuples: (image, label) of the training, validation dataset
    """
    (x_train, y_train), (x_test, y_test) = load_data(cache_dir=data_dir)

    # create the shards
    shard_num = int(shard_num)
    X_train = x_train[shard_num::collaborator_count]
    y_train = y_train[shard_num::collaborator_count]

    X_valid = x_test[shard_num::collaborator_count]
    y_valid = y_test[shard_num::collaborator_count]

    return (X_train, y_train), (X_valid, y_valid)


def load_mnist_shard(shard_num, collaborator_count, data_dir, categorical=True,
                     channels_last=True, **kwargs):
    """
    Load the MNIST dataset.

    Args:
        shard_num (int): The shard to use from the dataset
        collaborator_count (int): The number of collaborators in the federation
        categorical (bool): True = convert the labels to one-hot encoded
         vectors (Default = True)
        channels_last (bool): True = The input images have the channels
         last (Default = True)
        **kwargs: Additional parameters to pass to the function

    Returns:
        list: The input shape
        int: The number of classes
        numpy.ndarray: The training data
        numpy.ndarray: The training labels
        numpy.ndarray: The validation data
        numpy.ndarray: The validation labels
    """
    img_rows, img_cols = 28, 28
    num_classes = 10

    (X_train, y_train), (X_valid, y_valid) = _load_raw_datashards(
        data_dir, shard_num, collaborator_count
    )

    if channels_last:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    else:
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_valid /= 255

    logger.info(f'MNIST > X_train Shape : {X_train.shape}')
    logger.info(f'MNIST > y_train Shape : {y_train.shape}')
    logger.info(f'MNIST > Train Samples : {X_train.shape[0]}')
    logger.info(f'MNIST > Valid Samples : {X_valid.shape[0]}')

    if categorical:
        # convert class vectors to binary class matrices
        y_train = one_hot(y_train, num_classes)
        y_valid = one_hot(y_valid, num_classes)

    return input_shape, num_classes, X_train, y_train, X_valid, y_valid
