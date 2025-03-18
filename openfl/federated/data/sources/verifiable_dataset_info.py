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
        dataset_format: DatasetFormat,
        hash=None,
    ):
        self.data_sources = data_sources
        self.label = label
        self.metadata = metadata
        self.base_path = Path(base_path)
        self.dataset_format = dataset_format
        self.hash = hash
        if self.hash is None:
            self.hash = self.create_dataset_hash()

    def _create_verbose_dataset_hash(self):
        all_file_hashes = {
            str(file_path.relative_to(self.base_path)): ds.compute_object_hash(
                str(self.base_path / file_path)
            )
            for ds in self.data_sources
            for file_path in ds.enumerate_objects(str(self.base_path))
        }
        return all_file_hashes

    def _create_concise_dataset_hash(self):
        all_file_hashes = (
            ds.compute_object_hash(file)
            for ds in self.data_sources
            for file in ds.enumerate_objects(str(self.base_path))
        )
        sorted_file_hashes = sorted(all_file_hashes)
        joined_hashes = "".join(sorted_file_hashes)
        return sha384(joined_hashes.encode()).hexdigest()

    def create_dataset_hash(self):
        """Create the hash of the dataset according to dataset format."""
        if self.dataset_format == DatasetFormat.VERBOSE:
            return self._create_verbose_dataset_hash()
        if self.dataset_format == DatasetFormat.CONCISE:
            return self._create_concise_dataset_hash()
        raise ValueError("Unknown dataset format")

    def _validate_verbose_dataset_info(self):
        hashes = self._create_verbose_dataset_hash()
        return sorted(hashes.values()) == sorted(self.hash.values())

    def _validate_concise_dataset_info(self):
        concise_hash = self._create_concise_dataset_hash()
        return concise_hash == self.hash

    def verify_dataset(self, dataset_info=None):
        """Verify the dataset hash."""
        if dataset_info is None and self.hash is None:
            raise ValueError("No dataset info provided")
        if dataset_info:
            self.hash = dataset_info["hash"]

        if self.dataset_format == DatasetFormat.VERBOSE:
            return self._validate_verbose_dataset_info()
        if self.dataset_format == DatasetFormat.CONCISE:
            return self._validate_concise_dataset_info()
        raise ValueError("Unknown dataset format")

    def _verify_file_verbose(self, file_path, file_hash):
        if self.dataset_format != DatasetFormat.VERBOSE:
            raise ValueError("This method is only valid for verbose datasets")
        rel_file_path = Path(file_path).relative_to(Path(self.base_path))
        return self.hash[str(rel_file_path)] == file_hash

    def verify_single_file(self, file_path, file_hash):
        """Verify the hash of a single file."""
        if self.hash is None:
            raise ValueError("Hash not found in the dataset")
        if self.dataset_format == DatasetFormat.VERBOSE:
            return self._verify_file_verbose(file_path, file_hash)
        if self.dataset_format == DatasetFormat.CONCISE:
            raise ValueError("verify_single_file is only valid for verbose datasets")
        raise ValueError("Unknown dataset format")

    def to_json(self):
        """Serialize the VerifiableDatasetInfo to JSON"""
        dataset_dict = {
            "data_sources": [filter_non_serializable(ds) for ds in self.data_sources],
            "label": self.label,
            "metadata": self.metadata,
            "format": self.dataset_format.value,
        }
        dataset_dict["hash"] = self.create_dataset_hash()
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
            dataset_format=DatasetFormat(data_dict["format"]),
            hash=data_dict["hash"],
        )

    @staticmethod
    def deserialize_and_verify(json_str, base_path: Path):
        """Deserialize the VerifiableDatasetInfo from JSON and validate it."""
        data_dict = json.loads(json_str)
        vds = VerifiableDatasetInfo.from_dict(data_dict, base_path)
        return vds.verify_dataset(dataset_info=data_dict)
