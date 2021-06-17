# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import TensorFlowDataLoader
from .brats_utils import load_from_nifti


class TensorFlowBratsInMemory(TensorFlowDataLoader):
    """TensorFlow Data Loader for the BraTS dataset."""

    def __init__(self, data_path, batch_size, percent_train=0.8, pre_split_shuffle=True, **kwargs):
        """Initialize.

        Args:
            data_path: The file path for the BraTS dataset
            batch_size (int): The batch size to use
            percent_train (float): The percentage of the data to use for training (Default=0.8)
            pre_split_shuffle (bool): True= shuffle the dataset before
            performing the train/validate split (Default=True)
            **kwargs: Additional arguments, passed to super init and load_from_nifti

        Returns:
            Data loader with BraTS data
        """
        super().__init__(batch_size, **kwargs)

        X_train, y_train, X_valid, y_valid = load_from_nifti(parent_dir=data_path,
                                                             percent_train=percent_train,
                                                             shuffle=pre_split_shuffle,
                                                             **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
