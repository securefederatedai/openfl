# Copyright (C) 2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from os import path, makedirs
from collections.abc import Iterable

import numpy as np
from torch import manual_seed
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Grayscale, Compose
from torchvision.datasets import ImageFolder

from openfl.federated import PyTorchDataLoader

class TemplateDataLoader(PyTorchDataLoader):
    """Template dataloader for PyTorch.
    This class should be used as a template to create a custom DataLoader for your specific dataset.
    After generating this template, you should:
    1. Implement the `load_dataset` function to load your data.
    2. Modify the `plan.yaml` file to use this DataLoader.
    The `plan.yaml` modifications should be done under the `<workspace>/plan/plan.yaml` section:
    ```
    data_loader:
        defaults : plan/defaults/data_loader.yaml
        template: src.dataloader.TemplateDataLoader # Modify this line appropriately if you change the class name
        settings:
            # Add additional arguments (such as batch_size) that you wish to pass through `def __init__():`
            # You do not need to pass in data_path here. It will be set by the collaborators
    ```
    `batch_size` is passed to the `super().`__init__` method to ensure that the superclass is properly initialized with the specified batch size.
    After calling `super().__init__`, define `self.X_train`, `self.y_train`, `self.X_valid`, and `self.y_valid`.
    """

    def __init__(self, data_path, batch_size, **kwargs):
        """Initialize the data loader.
        Args:
            data_path: The file path to the data at the respective collaborator site.
            batch_size: The batch size of the data loader.
            **kwargs: Additional arguments that may be defined in `plan.yaml`
        """
        super().__init__(batch_size, **kwargs)

        # Load the dataset using the provided data_path and any additional kwargs.
        X_train, y_train, X_valid, y_valid = load_dataset(data_path, **kwargs)

        # Assign the loaded data to instance variables.
        self.X_train = X_train
        self.y_train = y_train
        self.train_loader = self.get_train_loader()

        self.X_valid = X_valid
        self.y_valid = y_valid
        self.val_loader = self.get_valid_loader()

def load_dataset(data_path, train_split_ratio=0.8, **kwargs):
    """
    Load your dataset here.
    This function should be implemented to load the dataset from the given `data_path`.
    You can use additional arguments passed via `**kwargs` if necessary.
    Args:
        data_path (str): Path to the data directory.
        **kwargs: Additional arguments that may be defined in `plan.yaml`
    Returns:
        Tuple containing:
        - numpy.ndarray: The training data.
        - numpy.ndarray: The training labels.
        - numpy.ndarray: The validation data.
        - numpy.ndarray: The validation labels.
    """
    # Implement dataset loading logic here and return the appropriate data.
    # Replace the following placeholders with actual data loading code.
    dataset = MNISTDataset(
        root=data_path,
        transform=Compose([Grayscale(num_output_channels=1), ToTensor()])
    )
    n_train = int(train_split_ratio * len(dataset))
    n_valid = len(dataset) - n_train

    ds_train, ds_val = random_split(
        dataset, lengths=[n_train, n_valid], generator=manual_seed(0))

    X_train, y_train = list(zip(*ds_train))

    X_train, y_train = np.stack(X_train), np.array(y_train)

    X_valid, y_valid = list(zip(*ds_val))
    X_valid, y_valid = np.stack(X_valid), np.array(y_valid)

    return X_train, y_train, X_valid, y_valid

class MNISTDataset(ImageFolder):
    """Encapsulates the MNIST dataset"""

    FOLDER_NAME = "mnist_images"
    DEFAULT_PATH = path.join(path.expanduser('~'), '.openfl', 'data')

    def __init__(self, root: str = DEFAULT_PATH, **kwargs) -> None:
        """Initialize."""
        makedirs(root, exist_ok=True)

        super(MNISTDataset, self).__init__(
            path.join(root, MNISTDataset.FOLDER_NAME), **kwargs)

    def __getitem__(self, index):
        """Allow getting items by slice index."""
        if isinstance(index, Iterable):
            return [super(MNISTDataset, self).__getitem__(i) for i in index]
        else:
            return super(MNISTDataset, self).__getitem__(index)
