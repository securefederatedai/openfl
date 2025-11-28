# Copyright (C) 2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between
# Intel Corporation and you.

import os

import modin.pandas as pd

from openfl.federated import XGBoostDataLoader


class HiggsDataLoader(XGBoostDataLoader):
    """
    DataLoader for the Higgs dataset.

    This class inherits from XGBoostDataLoader and is responsible for loading
    the Higgs dataset for training and validation.

    Attributes:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation features.
        y_valid (numpy.ndarray): Validation labels.
    """
    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)

        # Define default feature shape and number of classes for Higgs dataset
        self.feature_shape = (28,)
        self.num_classes = 2

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        X_train, y_train, X_valid, y_valid = load_Higgs(
            data_path, **kwargs
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

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


def load_Higgs(data_path, **kwargs):
    """
    Load the Higgs dataset from CSV files.

    The dataset is expected to be in two CSV files: 'train.csv' and 'valid.csv'.
    The first column in each file represents the labels, and the remaining
    columns represent the features.

    Args:
        data_path (str): The directory path where the CSV files are located.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing four elements:
            - X_train (numpy.ndarray): Training features.
            - y_train (numpy.ndarray): Training labels.
            - X_valid (numpy.ndarray): Validation features.
            - y_valid (numpy.ndarray): Validation labels.
    """
    train_data = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values

    valid_data = pd.read_csv(os.path.join(data_path, 'valid.csv'), header=None)
    X_valid = valid_data.iloc[:, 1:].values
    y_valid = valid_data.iloc[:, 0].values

    return X_train, y_train, X_valid, y_valid
