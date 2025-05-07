# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import KerasDataLoader

from .brats_utils import load_from_nifti


class KerasBratsInMemory(KerasDataLoader):
    """Keras Data Loader for the BraTS dataset."""

    def __init__(
        self, data_path=None, batch_size=32, percent_train=0.8, pre_split_shuffle=True, **kwargs
    ):
        """Initialize.

        Args:
            data_path: The file path for the BraTS dataset. If None, initialize for model
                creation only.
            batch_size (int): The batch size to use
            percent_train (float): The percentage of the data to use for training (Default=0.8)
            pre_split_shuffle (bool): True= shuffle the dataset before
            performing the train/validate split (Default=True)
            **kwargs: Additional arguments, passed to super init and load_from_nifti

        Returns:
            Data loader with BraTS data
        """
        super().__init__(batch_size, **kwargs)

        # Set default values for model initialization
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        X_train, y_train, X_valid, y_valid = load_from_nifti(parent_dir=data_path,
                                                             percent_train=percent_train,
                                                             shuffle=pre_split_shuffle,
                                                             **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
