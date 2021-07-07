# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""


import os

from openfl.federated import TensorFlowDataLoader
from .dataloader import DatasetGenerator


class TensorFlowBratsDataLoader(TensorFlowDataLoader):
    """TensorFlow Data Loader for the BraTS dataset."""

    def __init__(self, data_path, batch_size=4,
                 crop_dim=64, percent_train=0.8,
                 pre_split_shuffle=True,
                 number_input_channels=1,
                 num_classes=1,
                 **kwargs):
        """Initialize.

        Args:
            data_path: The file path for the BraTS dataset
            batch_size (int): The batch size to use
            crop_dim (int): Crop the original image to this size on each dimension
            percent_train (float): The percentage of the data to use for training (Default=0.8)
            pre_split_shuffle (bool): True= shuffle the dataset before
            performing the train/validate split (Default=True)
            **kwargs: Additional arguments, passed to super init

        Returns:
            Data loader with BraTS data
        """
        super().__init__(batch_size, **kwargs)

        self.data_path = os.path.abspath(os.path.expanduser(data_path))
        self.batch_size = batch_size
        self.crop_dim = [crop_dim, crop_dim, crop_dim, number_input_channels]
        self.num_input_channels = number_input_channels
        self.num_classes = num_classes

        self.train_test_split = percent_train

        self.brats_data = DatasetGenerator(crop_dim,
                                           data_path=data_path,
                                           number_input_channels=number_input_channels,
                                           batch_size=batch_size,
                                           train_test_split=percent_train,
                                           validate_test_split=0.5,
                                           num_classes=num_classes,
                                           random_seed=816)

    def get_feature_shape(self):
        """
        Get the shape of an example feature array.

        Returns:
            tuple: shape of an example feature array
        """
        return tuple(self.brats_data.get_input_shape())

    def get_train_loader(self, batch_size=None, num_batches=None):
        """
        Get training data loader.

        Returns
        -------
        loader object
        """
        return self.brats_data.ds_train

    def get_valid_loader(self, batch_size=None):
        """
        Get validation data loader.

        Returns:
            loader object
        """
        return self.brats_data.ds_val

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return self.brats_data.num_train

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        return self.brats_data.num_val
