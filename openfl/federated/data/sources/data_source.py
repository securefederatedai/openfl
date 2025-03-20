# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for different types of data sources."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator


class DataSourceType(Enum):
    """Enum for the different types of data sources."""

    LOCAL = "local"
    S3 = "s3"


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
        raise NotImplementedError

    @abstractmethod
    def enumerate_files(self, base_path: str) -> Generator[str, None, None]:
        """
        Enumerate all files in the data source.

        Args:
            base_path (str): the base path of the data source.

        Returns:
            list: A list of objects
        """
        yield

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
        raise NotImplementedError

    def _serialize_fields(self) -> Dict[str, Any]:
        """Returns a dictionary of serializable fields."""
        serializable_dict = {}
        for key, val in self.__dict__.items():
            if key.startswith("_"):  # Skip private attributes
                continue
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
