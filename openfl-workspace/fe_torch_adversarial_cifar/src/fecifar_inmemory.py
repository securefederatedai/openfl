# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import os
from pathlib import Path


import fastestimator as fe
import numpy as np
import tensorflow.keras.backend as K
from fastestimator.dataset import NumpyDataset
from fastestimator.op.numpyop.univariate import ChannelTranspose
from fastestimator.op.numpyop.univariate import Normalize
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file

from openfl.federated import FastEstimatorDataLoader


def load_data(cache_dir, image_key: str = 'x', label_key: str = 'y'):
    """Load and return the CIFAR10 dataset.

    Args:
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, eval_data)
    """
    cache_subdir = Path(cache_dir).expanduser().absolute()
    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(
        dirname,
        origin=origin,
        untar=True,
        file_hash='6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce',
        cache_subdir=cache_subdir)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)
    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    eval_data = NumpyDataset({image_key: x_test, label_key: y_test})
    return train_data, eval_data


class FastEstimatorCifarInMemory(FastEstimatorDataLoader):
    """TensorFlow Data Loader for MNIST Dataset."""

    def __init__(self, data_path, batch_size, collaborator_count, data_dir, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and
        #  what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size of
        # collaborator list.

        train_data, eval_data = load_data(data_dir)
        test_data = eval_data.split(0.5)

        train_data, eval_data, test_data = self.split_data(
            train_data,
            eval_data,
            test_data,
            int(data_path),
            collaborator_count
        )
        super().__init__(fe.Pipeline(
            train_data=train_data,
            eval_data=eval_data,
            test_data=test_data,
            batch_size=batch_size,
            ops=[
                Normalize(inputs='x', outputs='x',
                          mean=(0.4914, 0.4822, 0.4465),
                          std=(0.2471, 0.2435, 0.2616)),
                ChannelTranspose(inputs='x', outputs='x')
            ]), **kwargs)

        print(f'train_data = {train_data}')
        print(f'eval_data = {eval_data}')
        print(f'test_data = {test_data}')

        print(f'batch_size = {batch_size}')

    def split_data(self, train, eva, test, rank, collaborator_count):
        """Split data into N parts, where N is the collaborator count."""
        if collaborator_count == 1:
            return train, eva, test

        fraction = [1.0 / float(collaborator_count)]
        fraction *= (collaborator_count - 1)

        # Expand the split list into individual parameters
        train_split = train.split(*fraction)
        eva_split = eva.split(*fraction)
        test_split = test.split(*fraction)

        train = [train]
        eva = [eva]
        test = [test]

        if type(train_split) is not list:
            train.append(train_split)
            eva.append(eva_split)
            test.append(test_split)
        else:
            # Combine all partitions into a single list
            train = [train] + train_split
            eva = [eva] + eva_split
            test = [test] + test_split

        # Extract the right shard
        train = train[rank - 1]
        eva = eva[rank - 1]
        test = test[rank - 1]

        return train, eva, test
