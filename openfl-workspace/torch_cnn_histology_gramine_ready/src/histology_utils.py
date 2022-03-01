# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from collections.abc import Iterable
from logging import getLogger
from os import makedirs
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from tqdm import tqdm

from openfl.utilities import validate_file_hash

logger = getLogger(__name__)


class HistologyDataset(ImageFolder):
    """Colorectal Histology Dataset."""

    URL = ('https://zenodo.org/record/53169/files/Kather_'
           'texture_2016_image_tiles_5000.zip?download=1')
    FILENAME = 'Kather_texture_2016_image_tiles_5000.zip'
    FOLDER_NAME = 'Kather_texture_2016_image_tiles_5000'
    ZIP_SHA384 = ('7d86abe1d04e68b77c055820c2a4c582a1d25d2983e38ab724e'
                  'ac75affce8b7cb2cbf5ba68848dcfd9d84005d87d6790')
    DEFAULT_PATH = Path.cwd().absolute() / 'data'

    def __init__(self, root: Path = DEFAULT_PATH, **kwargs) -> None:
        """Initialize."""
        makedirs(root, exist_ok=True)
        filepath = root / HistologyDataset.FILENAME
        if not filepath.is_file():
            self.pbar = tqdm(total=None)
            urlretrieve(HistologyDataset.URL, filepath, self.report_hook)  # nosec
            validate_file_hash(filepath, HistologyDataset.ZIP_SHA384)
            with ZipFile(filepath, 'r') as f:
                f.extractall(root)

        super(HistologyDataset, self).__init__(root / HistologyDataset.FOLDER_NAME, **kwargs)

    def report_hook(self, count, block_size, total_size):
        """Update progressbar."""
        if self.pbar.total is None and total_size:
            self.pbar.total = total_size
        progress_bytes = count * block_size
        self.pbar.update(progress_bytes - self.pbar.n)

    def __getitem__(self, index):
        """Allow getting items by slice index."""
        if isinstance(index, Iterable):
            return [super(HistologyDataset, self).__getitem__(i) for i in index]
        else:
            return super(HistologyDataset, self).__getitem__(index)


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


def _load_raw_datashards(shard_num, collaborator_count, train_split_ratio=0.8):
    """
    Load the raw data by shard.

    Returns tuples of the dataset shard divided into training and validation.

    Args:
        shard_num (int): The shard number to use
        collaborator_count (int): The number of collaborators in the federation

    Returns:
        2 tuples: (image, label) of the training, validation dataset
    """
    dataset = HistologyDataset(transform=ToTensor())
    n_train = int(train_split_ratio * len(dataset))
    n_valid = len(dataset) - n_train
    ds_train, ds_val = random_split(
        dataset, lengths=[n_train, n_valid], generator=torch.manual_seed(0))

    # create the shards
    X_train, y_train = list(zip(*ds_train[shard_num::collaborator_count]))
    X_train, y_train = np.stack(X_train), np.array(y_train)

    X_valid, y_valid = list(zip(*ds_val[shard_num::collaborator_count]))
    X_valid, y_valid = np.stack(X_valid), np.array(y_valid)

    return (X_train, y_train), (X_valid, y_valid)


def load_histology_shard(shard_num, collaborator_count,
                         categorical=False, channels_last=False, **kwargs):
    """
    Load the Histology dataset.

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
    img_rows, img_cols = 150, 150
    num_classes = 8

    (X_train, y_train), (X_valid, y_valid) = _load_raw_datashards(
        shard_num, collaborator_count)

    if channels_last:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    else:
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)

    logger.info(f'Histology > X_train Shape : {X_train.shape}')
    logger.info(f'Histology > y_train Shape : {y_train.shape}')
    logger.info(f'Histology > Train Samples : {X_train.shape[0]}')
    logger.info(f'Histology > Valid Samples : {X_valid.shape[0]}')

    if categorical:
        # convert class vectors to binary class matrices
        y_train = one_hot(y_train, num_classes)
        y_valid = one_hot(y_valid, num_classes)

    return input_shape, num_classes, X_train, y_train, X_valid, y_valid
