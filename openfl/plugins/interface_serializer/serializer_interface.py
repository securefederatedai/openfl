# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Serializer plugin interface."""


class Serializer:
    """Serializer API plugin."""

    def __init__(self) -> None:
        """Initialize serializer."""
        pass

    @staticmethod
    def serialize(object_, filename):
        """Serialize an object and save to disk.

        This is a static method that is not implemented.

        Args:
            object_ (object): The object to be serialized.
            filename (str): The name of the file where the serialized object
                will be saved.

        Returns:
            None

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def restore_object(filename):
        """Load and deserialize an object.

        This is a static method that is not implemented.

        Args:
            filename (str): The name of the file where the serialized object
                is saved.

        Returns:
            object: The deserialized object.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError
