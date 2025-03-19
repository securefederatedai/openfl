# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for different types of data sources."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generator


class DataSourceType(Enum):
    """Enum for the different types of data sources."""

    LOCAL = "local"
    S3 = "s3"


class DataSource(ABC):
    """
    Base class for different types of data sources.

    Attributes:
        datasource_type (str): The storage type of the data source
    """

    def __init__(self, datasource_type: DataSourceType):
        """
        Initialize a DataSource.

        Args:
            datasource_type (DataSourceType): The storage type of the data source.
        """
        self.datasource_type = datasource_type

    @abstractmethod
    def compute_object_hash(self, path: str) -> str:
        """
        Compute the hash of the object or file.

        Args:
            path (str): Path to the file.

        Returns:
            str: The file's hash.
        """
        raise NotImplementedError

    @abstractmethod
    def enumerate_objects(self, base_path: str) -> Generator[str, None, None]:
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
