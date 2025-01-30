# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""DataLoader module."""


class DataLoader:
    """A base class used to represent a Federated Learning Data Loader.

    This class should be inherited by any data loader class specific to a
    machine learning framework.

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        """Initializes the DataLoader object.

        Args:
            kwargs: Additional arguments to pass to the function.
        """
        pass

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Raises:
            NotImplementedError: This method must be implemented by a child
                class.
        """
        raise NotImplementedError

    def get_train_loader(self, **kwargs):
        """Returns the data loader for the training data.

        Args:
            kwargs: Additional arguments to pass to the function.

        Raises:
            NotImplementedError: This method must be implemented by a child
                class.
        """
        raise NotImplementedError

    def get_valid_loader(self):
        """Returns the data loader for the validation data.

        Raises:
            NotImplementedError: This method must be implemented by a child
                class.
        """
        raise NotImplementedError

    def get_infer_loader(self):
        """Returns the data loader for inferencing data.

        Raises:
            NotImplementedError: This method must be implemented by a child
                class.
        """
        return NotImplementedError

    def get_train_data_size(self):
        """Returns the total number of training samples.

        Raises:
            NotImplementedError: This method must be implemented by a child
                class.
        """
        raise NotImplementedError

    def get_valid_data_size(self):
        """Returns the total number of validation samples.

        Raises:
            NotImplementedError: This method must be implemented by a child
                class.
        """
        raise NotImplementedError
