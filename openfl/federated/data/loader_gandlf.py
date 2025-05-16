# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""PyTorchDataLoader module."""

from openfl.federated.data.loader import DataLoader


class GaNDLFDataLoaderWrapper(DataLoader):
    """A class used to represent a data loader for the Generally Nuanced Deep
    Learning Framework (GaNDLF).

    Attributes:
        train_csv (str): Path to the training CSV file.
        val_csv (str): Path to the validation CSV file.
        train_dataloader (DataLoader): DataLoader object for the training data.
        val_dataloader (DataLoader): DataLoader object for the validation data.
        feature_shape (tuple): Shape of an example feature array.
    """

    def __init__(self, data_path=None, **kwargs):
        """Initializes the GaNDLFDataLoaderWrapper object.

        Args:
            data_path (str, optional): The path to the directory containing the data.
                If None, initialize for model creation only.
            **kwargs: Additional arguments to pass to the function.
        """
        self.train_csv = None
        self.val_csv = None
        self.train_dataloader = None
        self.val_dataloader = None

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        # Otherwise set up paths for actual data loading
        if "inference" in data_path:
            self.train_csv = None
        else:
            self.train_csv = data_path + "/train.csv"
        self.val_csv = data_path + "/valid.csv"

    def set_dataloaders(self, train_dataloader, val_dataloader):
        """Sets the data loaders for the training and validation data.

        Args:
            train_dataloader (DataLoader): The DataLoader object for the
                training data.
            val_dataloader (DataLoader): The DataLoader object for the
                validation data.
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Returns:
            tuple: The shape of an example feature array.
                derives shape from the GANDLF config's patch_size if available
        """

        # If we have a train dataloader with a dataset that has a gandlf_params attribute
        # with patch_size
        if (
            self.train_dataloader is not None
            and hasattr(self.train_dataloader, "dataset")
            and hasattr(self.train_dataloader.dataset, "gandlf_params")
            and "patch_size" in self.train_dataloader.dataset.gandlf_params
        ):
            # Return the patch size from GANDLF config
            return self.train_dataloader.dataset.gandlf_params["patch_size"]

        raise NotImplementedError(
            "Ensure gandlf params have to be defined in gandlf config file"
            " or implement get_feature_shape method in the GaNDLFDataLoaderWrapper class."
        )

    def get_train_loader(self, batch_size=None, num_batches=None):
        """Returns the data loader for the training data.

        Args:
            batch_size (int, optional): The batch size for the data loader
                (default is None).
            num_batches (int, optional): The number of batches for the data
                loader (default is None).

        Returns:
            DataLoader: The DataLoader object for the training data.
        """
        return self.train_dataloader

    def get_valid_loader(self, batch_size=None):
        """Returns the data loader for the validation data.

        Args:
            batch_size (int, optional): The batch size for the data loader
                (default is None).

        Returns:
            DataLoader: The DataLoader object for the validation data.
        """
        return self.val_dataloader

    def get_train_data_size(self):
        """Returns the total number of training samples.

        Returns:
            int: The total number of training samples or 0 if not loaded.
        """
        if self.train_dataloader is None or not hasattr(self.train_dataloader, "dataset"):
            return 0
        return len(self.train_dataloader.dataset)

    def get_valid_data_size(self):
        """Returns the total number of validation samples.

        Returns:
            int: The total number of validation samples or 0 if not loaded.
        """
        if self.val_dataloader is None or not hasattr(self.val_dataloader, "dataset"):
            return 0
        return len(self.val_dataloader.dataset)
