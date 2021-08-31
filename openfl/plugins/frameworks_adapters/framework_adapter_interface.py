# Copyright (C) 2020-2021 Intel Corporation
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
        """
        Extract tensor dict from a model and an optimizer.

        Returns:
        dict {weight name: numpy ndarray}
        """
        raise NotImplementedError

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        """
        Set tensor dict from a model and an optimizer.

        Given a dict {weight name: numpy ndarray} sets weights to
        the model and optimizer objects inplace.
        """
        raise NotImplementedError
