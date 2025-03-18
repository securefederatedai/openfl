# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""This module contains the LocalDataSource class."""

from hashlib import sha384
from pathlib import Path

from openfl.federated.data.sources.data_source import DataSource, DataSourceType


class LocalDataSource(DataSource):
    """This class represents a local data source."""

    def __init__(self, source_path: Path, hash_func=sha384, max_dataset_size=0):
        super().__init__(DataSourceType.LOCAL)
        self.source_path = Path(source_path)
        self.hash_func = hash_func
        self.max_dataset_size = max_dataset_size

    def enumerate_objects(self, base_path: str):
        """Enumerate all files in the data source."""
        total_size_bytes = 0
        full_path = Path(base_path) / self.source_path
        if full_path.is_dir():
            for file_path in full_path.glob("**/*.*"):
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

    def compute_object_hash(self, path: str) -> str:
        """Compute the hash of the file. Return hash on hexstring format."""
        hash_obj = self.hash_func()
        with open(path, "rb") as file:
            for byte_block in iter(lambda: file.read(65536), b""):
                hash_obj.update(byte_block)
            return hash_obj.hexdigest()

    @classmethod
    def from_dict(cls, ds_dict: dict):
        return cls(source_path=Path(ds_dict["source_path"]))
