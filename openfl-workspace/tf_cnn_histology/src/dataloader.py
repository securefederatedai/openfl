# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import TensorFlowDataLoader
from logging import getLogger

import numpy as np
import tensorflow_datasets as tfds

logger = getLogger(__name__)


class HistologyDataloader(TensorFlowDataLoader):
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


def _load_raw_datashards(shard_num, collaborator_count):
    """
    Load the raw data by shard.

    Returns tuples of the dataset shard divided into training and validation.

    Args:
        shard_num (int): The shard number to use
        collaborator_count (int): The number of collaborators in the federation

    Returns:
        2 tuples: (image, label) of the training, validation dataset
    """
    (ds), metadata = tfds.load('colorectal_histology', data_dir='.',
                               shuffle_files=False, split='train', batch_size=-1,
                               with_info=True, as_supervised=True)

    image, label = tfds.as_numpy(ds)

    np.random.seed(42)
    shuf = np.random.permutation(len(image))
    image = image[shuf]
    label = label[shuf]

    split = int(len(image) * 0.8)

    X_train_tot = image[:split]
    y_train_tot = label[:split]

    X_valid_tot = image[split:]
    y_valid_tot = label[split:]

    shard_num = int(shard_num)

    # create the shards
    X_train = X_train_tot[shard_num::collaborator_count]
    y_train = y_train_tot[shard_num::collaborator_count]

    X_valid = X_valid_tot[shard_num::collaborator_count]
    y_valid = y_valid_tot[shard_num::collaborator_count]

    return (X_train, y_train), (X_valid, y_valid)


def load_histology_shard(shard_num, collaborator_count, categorical=True,
                         channels_last=True, **kwargs):
    """
    Load the colorectal histology dataset.

    Args:
        shard_num (int): The shard to use from the dataset
        collaborator_count (int): The number of collaborators in the federation
        categorical (bool): True = convert the labels to one-hot encoded
         vectors (Default = True)
        channels_last (bool): True = The input images have the channels last
         (Default = True)
        **kwargs: Additional parameters to pass to the function

    Returns:
        list: The input shape
        int: The number of classes
        numpy.ndarray: The training data
        numpy.ndarray: The training labels
        numpy.ndarray: The validation data
        numpy.ndarray: The validation labels
    """
    num_classes = 8
    img_rows = 150
    img_cols = 150
    channels = 3

    (X_train, y_train), (X_valid, y_valid) = _load_raw_datashards(
        shard_num, collaborator_count)

    if channels_last:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)
    else:
        X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_valid /= 255

    logger.info(f'Histology > X_train Shape : {X_train.shape}')
    logger.info(f'Histology > y_train Shape : {y_train.shape}')
    logger.info(f'Histology > Train Samples : {X_train.shape[0]}')
    logger.info(f'Histology > Valid Samples : {X_valid.shape[0]}')

    if categorical:
        # convert class vectors to binary class matrices
        y_train = np.eye(num_classes)[y_train]
        y_valid = np.eye(num_classes)[y_valid]

    return input_shape, num_classes, X_train, y_train, X_valid, y_valid
