# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module contains the VerifiableDatasetInfo class."""

import json
from enum import Enum
from hashlib import sha384
from pathlib import Path
from typing import List

from openfl.federated.data.sources.data_source import DataSource, DataSourceType
from openfl.federated.data.sources.local_data_source import LocalDataSource


class DatasetFormat(Enum):
    """Enum for the different dataset commitment formats."""

    VERBOSE = "verbose_dataset"
    CONCISE = "concise_dataset"


def filter_non_serializable(obj):
    """Filter out methods and non-serializable objects."""
    serializable_dict = {}
    for key, val in obj.__dict__.items():
        if callable(val):
            continue  # Skip methods
        if isinstance(val, Enum):
            val = val.value  # Convert Enum to its value
        elif isinstance(val, Path):
            val = str(val)  # Convert Path to string
        serializable_dict[key] = val
    return serializable_dict


class VerifiableDatasetInfo:
    """
    This class represents a data set whose integrity can be verified.
    It contains multiple data sources and methods to compute data commitments, verification,
    as well as utilities for serialization and deserialization.
    """

    def __init__(
        self,
        data_sources: List[DataSource],
        base_path: Path,
        label: str,
        metadata,
        root_hash=None,
    ):
        self.data_sources = data_sources
        self.label = label
        self.metadata = metadata
        self.base_path = Path(base_path)
        self.root_hash = root_hash
        self.all_hashes = None
        if self.root_hash is None:
            self.root_hash = self.create_dataset_hash()

    def _create_verbose_dataset_hash(self):
        self.all_hashes = {
            str(file_path.relative_to(self.base_path)): ds.compute_object_hash(
                str(self.base_path / file_path)
            )
            for ds in self.data_sources
            for file_path in ds.enumerate_objects(str(self.base_path))
        }
        return self.all_hashes

    def _create_concise_dataset_hash(self):
        all_file_hashes = self._create_verbose_dataset_hash()
        sorted_file_hashes = sorted(all_file_hashes.values())
        joined_hashes = "".join(sorted_file_hashes)
        self.root_hash = sha384(joined_hashes.encode()).hexdigest()
        return self.root_hash

    def create_dataset_hash(self):
        """Create and return the root_hash of all files hashes."""
        return self._create_concise_dataset_hash()

    def _validate_verbose_dataset_info(self):
        hashes = self._create_verbose_dataset_hash()
        return sorted(hashes.values()) == sorted(self.all_hashes.values())

    def _validate_concise_dataset_info(self):
        concise_hash = self._create_concise_dataset_hash()
        return concise_hash == self.root_hash

    def verify_dataset(self, dataset_info=None):
        """Verify the dataset root_hash."""
        if dataset_info is None and self.root_hash is None:
            raise ValueError("No dataset info provided")
        if dataset_info:
            self.root_hash = dataset_info["root_hash"]
        return self._validate_concise_dataset_info()

    def _verify_file_verbose(self, file_path, file_hash):
        rel_file_path = Path(file_path).relative_to(Path(self.base_path))
        return self.all_hashes[str(rel_file_path)] == file_hash

    def verify_single_file(self, file_path, file_hash):
        """Verify the hash of a single file."""
        if self.all_hashes is None:
            raise ValueError("Files hashes not found in the dataset")
        return self._verify_file_verbose(file_path, file_hash)

    def to_json(self):
        """Serialize the VerifiableDatasetInfo to JSON"""
        dataset_dict = {
            "data_sources": [filter_non_serializable(ds) for ds in self.data_sources],
            "label": self.label,
            "metadata": self.metadata,
        }
        dataset_dict["root_hash"] = self.create_dataset_hash()
        return json.dumps(dataset_dict, sort_keys=True, indent=4)

    @staticmethod
    def from_dict(data_dict, base_path: Path):
        """Deserialize the VerifiableDatasetInfo from JSON"""

        # Create appropriate data source based on dictionary information
        data_sources = []
        for datasource in data_dict["data_sources"]:
            if datasource["datasource_type"] == DataSourceType.LOCAL.value:
                data_source = LocalDataSource.from_dict(ds_dict=datasource)
            # elif datasource['datasource_type'] == DataSourceType.S3.value:
            #     data_source = S3DataSource.from_dict(ds_dict=datasource)
            else:
                raise ValueError(f"Unknown storage type: {datasource['datasource_type']}")
            data_sources.append(data_source)

        return VerifiableDatasetInfo(
            data_sources,
            base_path=base_path,
            label=data_dict["label"],
            metadata=data_dict["metadata"],
            root_hash=data_dict["root_hash"],
        )

    @staticmethod
    def deserialize_and_verify(json_str, base_path: Path):
        """Deserialize the VerifiableDatasetInfo from JSON and validate it."""
        data_dict = json.loads(json_str)
        vds = VerifiableDatasetInfo.from_dict(data_dict, base_path)
        return vds.verify_dataset(dataset_info=data_dict)
