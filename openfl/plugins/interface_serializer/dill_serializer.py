"""Dill serializer plugin."""

import dill  # nosec

from openfl.plugins.interface_serializer.serializer_interface import Serializer


class DillSerializer(Serializer):
    """Serializer API plugin."""

    def __init__(self) -> None:
        """Initialize serializer."""
        super().__init__()

    @staticmethod
    def serialize(object_, filename):
        """Serialize an object and save to disk."""
        with open(filename, 'wb') as f:
            dill.dump(object_, f, recurse=True)

    @staticmethod
    def restore_object(filename):
        """Load and deserialize an object."""
        with open(filename, 'rb') as f:
            return dill.load(f)  # nosec
