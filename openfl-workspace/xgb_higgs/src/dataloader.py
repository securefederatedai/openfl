# Copyright (C) 2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between
# Intel Corporation and you.

from openfl.federated import XGBoostDataLoader
import os
import modin.pandas as pd

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
    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        X_train, y_train, X_valid, y_valid = load_Higgs(
            data_path, **kwargs
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid


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
