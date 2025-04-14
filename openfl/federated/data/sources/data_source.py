# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for different types of data sources."""

import hashlib
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Generator


class DataSourceType(Enum):
    """Enum for the different types of data sources."""

    LOCAL = auto()
    S3 = auto()


class DataSource(ABC):
    """
    Base class for different types of data sources.

    Attributes:
        type (str): The storage type of the data source
    """

    def __init__(self, type: DataSourceType):
        """
        Initialize a DataSource.

        Args:
            type (DataSourceType): The storage type of the data source.
        """
        self.type = type

    @abstractmethod
    def compute_file_hash(self, path: str) -> str:
        """
        Compute the hash of the object or file.

        Args:
            path (str): Path to the file.

        Returns:
            str: The file's hash.
        """
        pass

    @abstractmethod
    def enumerate_files(self) -> Generator[str, None, None]:
        """
        Enumerate all files in the data source.

        Returns:
            list: A list of objects
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, ds_dict: dict):
        """
        Create a DataSource from a dictionary.

        Args:
            ds_dict (dict): The dictionary to convert.

        Returns:
            DataSource: The created DataSource.
        """
        pass

    def is_valid_hash_function(self, func):
        try:
            func_name = func.__name__.removeprefix("openssl_")

            # Ensure it is a known hashlib function
            if func_name not in hashlib.algorithms_available:
                return False

            # Ensure it's callable and returns a valid hash object
            test_instance = func()
            return callable(func) and hasattr(test_instance, "digest")

        except Exception:
            return False

    def _serialize_fields(self) -> Dict[str, Any]:
        """Returns a dictionary of serializable fields."""
        serializable_dict = {}
        for key, val in self.__dict__.items():
            if key.startswith("_"):  # Skip private attributes
                continue
            if self.is_valid_hash_function(val):
                val = val.__name__.removeprefix("openssl_")
            if callable(val):
                continue  # Skip methods
            if isinstance(val, Path):
                val = str(val)  # Convert Path to string
            elif isinstance(val, Enum):
                val = val.value  # Convert Enum to its value
            serializable_dict[key] = val
        return serializable_dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary using the serialization rules."""
        return self._serialize_fields()
