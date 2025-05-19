# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from collections.abc import Iterable
from logging import getLogger
import os
import sys


from openfl.federated import PyTorchDataLoader
import numpy as np
from openfl.federated.data.sources.torch.verifiable_map_style_image_folder import VerifiableImageFolder
from openfl.federated.data.sources.data_sources_json_parser import DataSourcesJsonParser
from openfl.utilities.path_check import is_directory_traversal
import torch
from torch.utils.data import random_split
from torchvision.transforms import ToTensor


logger = getLogger(__name__)


class PyTorchHistologyVerifiableDataLoader(PyTorchDataLoader):
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

        # Set Histology-specific default attributes
        self.feature_shape = [3, 150, 150]
        self.num_classes = 8

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            logger.info("Initializing dataloader for model creation only (no data loading)")
            return

        verifible_dataset_info = self.get_verifiable_dataset_info(data_path)

        hash_file_path = os.path.join(data_path, "hash.txt")
        verify_dataset_items = False
        if os.path.isfile(hash_file_path):
            logger.info(f"Found hash file at: {hash_file_path}. Verifying dataset...")
            verify_dataset_items = True
            with open(hash_file_path, "r", encoding="utf-8") as file:
                hash_value = file.read()
            dataset_valid = verifible_dataset_info.verify_dataset(root_hash=hash_value)
            if not dataset_valid:
                logger.error("The dataset is not valid.")
                sys.exit(1)
            else:
                logger.info("The dataset is valid.")

        _, num_classes, X_train, y_train, X_valid, y_valid = load_histology_shard(
            verifible_dataset_info=verifible_dataset_info, verify_dataset_items=verify_dataset_items, **kwargs)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes


    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Returns:
            list: The shape of an example feature array [3, 150, 150] for Histology images.
        """
        return self.feature_shape

    def get_num_classes(self):
        """Returns the number of classes for classification tasks.

        Returns:
            int: The number of classes (8 for Histology dataset).
        """
        return self.num_classes

    def get_verifiable_dataset_info(self, data_path):
        """
        Parse dataset information from `datasources.json` in the given directory.

        Args:
            data_path (str): Path to the directory containing `datasources.json`.

        Returns:
            VerifiableDatasetInfo: Parsed dataset information.

        Raises:
            SystemExit: If `data_path` is invalid or missing `datasources.json`.
        """
        """Return the verifiable dataset info object for the given data sources."""
        if data_path and is_directory_traversal(data_path):
            logger.error("Data path is out of the openfl workspace scope.")
        if not os.path.isdir(data_path):
            logger.error("The data path must be a directory.")
            sys.exit(1)

        datasources_json_path = os.path.join(data_path, "datasources.json")
        if not os.path.isfile(datasources_json_path):
            logger.error("The directory must contain a file named 'datasources.json' at the first level.")
            sys.exit(1)
        with open(datasources_json_path, "r", encoding="utf-8") as file:
            data = file.read()
        return DataSourcesJsonParser.parse(data)


class HistologyDataset(VerifiableImageFolder):
    """Colorectal Histology Dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize."""
        super().__init__(**kwargs)

    def __getitem__(self, index):
        """Allow getting items by slice index."""
        if isinstance(index, Iterable):
            return [super().__getitem__(i) for i in index]
        else:
            return super().__getitem__(index)


def _load_raw_data(verifiable_dataset_info, verify_dataset_items=False, train_split_ratio=0.8, **kwargs):
    """
    Load the raw data by shard.

    Returns tuples of the dataset shard divided into training and validation.

    Args:
        data_path (str): The path to the dataset.

    Returns:
        2 tuples: (image, label) of the training, validation dataset
    """
    dataset = HistologyDataset(
        vds=verifiable_dataset_info,
        transform=ToTensor(),
        verify_dataset_items=verify_dataset_items
    )
    n_train = int(train_split_ratio * len(dataset))
    n_valid = len(dataset) - n_train
    ds_train, ds_val = random_split(
        dataset, lengths=[n_train, n_valid], generator=torch.manual_seed(0))

    # create the shards
    X_train, y_train = list(zip(*ds_train))
    X_train, y_train = np.stack(X_train), np.array(y_train)

    X_valid, y_valid = list(zip(*ds_val))
    X_valid, y_valid = np.stack(X_valid), np.array(y_valid)

    return (X_train, y_train), (X_valid, y_valid)



def load_histology_shard(verifible_dataset_info, verify_dataset_items,
                         categorical=False, channels_last=False, **kwargs):
    """
    Load the Histology dataset.

    Args:
        data_path (str): path to data directory
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

    (X_train, y_train), (X_valid, y_valid) = _load_raw_data(verifible_dataset_info, verify_dataset_items, **kwargs)

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
        y_train = np.eye(num_classes)[y_train]
        y_valid = np.eye(num_classes)[y_valid]

    return input_shape, num_classes, X_train, y_train, X_valid, y_valid
