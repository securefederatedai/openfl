# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DataLoader module."""


class DataLoader(object):
    """Federated Learning Data Loader Class."""

    def __init__(self, **kwargs):
        """
        Instantiate the data object.

        Returns:
            None
        """
        pass

    def get_feature_shape(self):
        """
        Get the shape of an example feature array.

        Returns:
            tuple: shape of an example feature array
        """
        raise NotImplementedError

    def get_train_loader(self, **kwargs):
        """
        Get training data loader.

        Returns:
            loader object (class defined by inheritor)
        """
        raise NotImplementedError

    def get_valid_loader(self):
        """
        Get validation data loader.

        Returns:
            loader object (class defined by inheritor)
        """
        raise NotImplementedError

    def get_infer_loader(self):
        """
        Get inferencing data loader.

        Returns
        -------
        loader object (class defined by inheritor)
        """
        return NotImplementedError

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        raise NotImplementedError

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        raise NotImplementedError
