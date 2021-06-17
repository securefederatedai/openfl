# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Serializer plugin interface."""


class Serializer:
    """Serializer API plugin."""

    def __init__(self) -> None:
        """Initialize serializer."""
        pass

    @staticmethod
    def serialize(object_, filename):
        """Serialize an object and save to disk."""
        raise NotImplementedError

    @staticmethod
    def restore_object(filename):
        """Load and deserialize an object."""
        raise NotImplementedError
