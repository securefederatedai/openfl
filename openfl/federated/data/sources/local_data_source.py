# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""This module contains the LocalDataSource class."""

import hashlib
import os
from hashlib import sha384
from pathlib import Path
from typing import Callable, Generator

from openfl.federated.data.sources.data_source import DataSource, DataSourceType


class LocalDataSource(DataSource):
    """This class represents a local data source."""

    def __init__(
        self,
        source_path: Path,
        base_path,
        hash_func: Callable[..., "hashlib._Hash"] = sha384,
        max_dataset_size=0,
    ):
        """
        Initialize a LocalDataSource object.

        Args:
            source_path (Path): The path to the source data, relative to base_path.
            base_path (Path): The base path to the data source.
            hash_func (Callable[..., hashlib._Hash]): The hash function from hashlib
            to use to hash the data.
            max_dataset_size (int): The maximum size of the dataset in GB.
        """
        super().__init__(DataSourceType.LOCAL)
        self.source_path = Path(source_path)
        if not super().is_valid_hash_function(hash_func):
            raise ValueError(
                f"Invalid hash function: {hash_func.__name__}. Must be a hashlib function."
            )
        self.hash_func = hash_func
        self.max_dataset_size = max_dataset_size
        self._base_path = Path(base_path)  # private attribute, will not be serialized

    def get_source_full_path(self):
        """Return the full path to the source data."""
        return self._base_path / self.source_path

    def enumerate_files(self) -> Generator[str, None, None]:
        """Enumerate all files in the data source."""
        total_size_bytes = 0
        full_path = Path(self._base_path) / self.source_path
        if full_path.is_dir():
            for root, _, files in os.walk(full_path):
                for file in files:
                    file_path = Path(root) / file
                    if self.max_dataset_size > 0:
                        total_size_bytes += file_path.stat().st_size
                        total_size_gb = total_size_bytes / (1024**3)
                        if total_size_gb > self.max_dataset_size:
                            raise ValueError(
                                f"Total dataset size: {total_size_gb:.2f} GB exceeds "
                                f"{self.max_dataset_size} GB"
                            )
                    yield file_path

        elif full_path.is_file():
            if self.max_dataset_size > 0:
                total_size_bytes = full_path.stat().st_size
                total_size_gb = total_size_bytes / (1024**3)
                if total_size_gb > self.max_dataset_size:
                    raise ValueError(
                        f"Total dataset size: {total_size_gb:.2f} GB exceeds "
                        f"{self.max_dataset_size} GB"
                    )
            yield full_path

    def compute_file_hash(self, path: str) -> str:
        """Compute the hash of the file. Return hash on hexstring format."""
        hash_obj = self.hash_func()
        with open(path, "rb") as file:
            for byte_block in iter(lambda: file.read(65536), b""):
                hash_obj.update(byte_block)
        return hash_obj.hexdigest()

    @classmethod
    def from_dict(cls, ds_dict: dict, base_path):
        source_path = Path(ds_dict["source_path"])
        # Retrieve function from hashlib
        hash_func = getattr(hashlib, ds_dict.get("hash_func", "sha384"), None)
        max_dataset_size = ds_dict.get("max_dataset_size", 0)
        return cls(
            source_path=source_path,
            base_path=base_path,
            hash_func=hash_func,
            max_dataset_size=max_dataset_size,
        )
