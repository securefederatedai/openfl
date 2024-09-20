# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Framework Adapter plugin interface."""


class FrameworkAdapterPluginInterface:
    """Framework adapter plugin class."""

    def __init__(self) -> None:
        """Initialize framework adapter."""
        pass

    @staticmethod
    def serialization_setup():
        """Prepare model for serialization (optional)."""
        pass

    @staticmethod
    def get_tensor_dict(model, optimizer=None) -> dict:
        """Extract tensor dict from a model and an optimizer.

        Args:
            model (object): The model object.
            optimizer (object, optional): The optimizer object. Defaults to
                None.

        Returns:
            dict: A dictionary with weight name as key and numpy ndarray as
                value.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device="cpu"):
        """
        Set tensor dict from a model and an optimizer.

        Given a dict {weight name: numpy ndarray} sets weights to
        the model and optimizer objects inplace.

        Args:
            model (object): The model object.
            tensor_dict (dict): Dictionary with weight name as key and numpy
                ndarray as value.
            optimizer (object, optional): The optimizer object. Defaults to
                None.
            device (str, optional): The device to be used. Defaults to 'cpu'.

        Returns:
            None

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError
