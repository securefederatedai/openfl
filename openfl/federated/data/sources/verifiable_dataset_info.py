# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module contains the VerifiableDatasetInfo class."""

import json
from hashlib import sha384
from typing import List

from openfl.federated.data.sources.data_source import DataSource, DataSourceType
from openfl.federated.data.sources.local_data_source import LocalDataSource
from openfl.federated.data.sources.s3_data_source import S3DataSource


class VerifiableDatasetInfo:
    """
    This class represents a data set whose integrity can be verified.
    It contains multiple data sources and methods to compute data commitments, verification,
    as well as utilities for serialization and deserialization.
    """

    def __init__(
        self,
        data_sources: List[DataSource],
        label: str,
        metadata=None,
        root_hash=None,
    ):
        self.data_sources = data_sources
        self.label = label
        self.metadata = metadata
        self.root_hash = root_hash
        self.all_hashes = None

    def _create_verbose_dataset_hash(self):
        all_hashes = {
            str(file_path): ds.compute_file_hash(str(file_path))
            for ds in self.data_sources
            for file_path in ds.enumerate_files()
        }
        return all_hashes

    def _create_concise_dataset_hash(self):
        all_file_hashes = self._create_verbose_dataset_hash()
        sorted_file_hashes = sorted(all_file_hashes.values())
        joined_hashes = "".join(sorted_file_hashes)
        root_hash = sha384(joined_hashes.encode()).hexdigest()
        return root_hash, all_file_hashes

    def create_dataset_hash(self):
        """Create and return the root_hash of all files hashes."""
        root_hash, _ = self._create_concise_dataset_hash()
        return root_hash

    def _validate_verbose_dataset_info(self):
        hashes = self._create_verbose_dataset_hash()
        return sorted(hashes.values()) == sorted(self.all_hashes.values())

    def _validate_concise_dataset_info(self):
        concise_hash, all_hashes = self._create_concise_dataset_hash()
        valid = concise_hash == self.root_hash
        if valid and self.all_hashes is None:
            self.all_hashes = all_hashes
        return valid

    def verify_dataset(self, root_hash=None):
        """Verify the dataset against root_hash."""
        if root_hash is None and self.root_hash is None:
            raise ValueError("No saved root hash found. Please provide 'dataset_info'.")
        if root_hash:
            self.root_hash = root_hash
        return self._validate_concise_dataset_info()

    def _verify_file_verbose(self, file_path, file_hash):
        if str(file_path) not in self.all_hashes:
            raise KeyError(f"Verification failed: No information found for the file: {file_path}")
        return self.all_hashes[str(file_path)] == file_hash

    def verify_single_file(self, file_path, file_hash):
        """Verify the hash of a single file."""
        if self.all_hashes is None:
            if self.root_hash is not None:
                self.verify_dataset(self.root_hash)
            else:
                raise ValueError("Trusted hash not found in the dataset")
        return self._verify_file_verbose(file_path, file_hash)

    def to_json(self):
        if len(self.data_sources) == 1:
            return self._to_json_v1()
        return self._to_json_v2()

    def _to_json_v1(self):
        if self.data_sources[0].type == DataSourceType.LOCAL:
            path = str(self.data_sources[0].get_source_full_path())
        elif self.data_sources[0].type == DataSourceType.S3:
            path = self.data_sources[0].uri
        else:
            raise ValueError(f"Unknown storage type: {self.data_sources[0].type}")

        dataset_dict = {
            "dataset_id": self.create_dataset_hash(),
            "mount_absolute_path": path,
            "label": self.label,
            "metadata": self.metadata,
            "dataset_format": "concise_dataset",
        }
        if self.data_sources[0].type == DataSourceType.S3:
            s3_dict = self.data_sources[0].to_dict()
            s3_dict.pop("uri")
            dataset_dict.update(s3_dict)

        return json.dumps(dataset_dict, sort_keys=True, indent=4)

    def _to_json_v2(self):
        dataset_dict = {
            "version": "2.0",
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "label": self.label,
            "metadata": self.metadata,
        }
        dataset_dict["root_hash"] = self.create_dataset_hash()
        return json.dumps(dataset_dict, sort_keys=True, indent=4)

    def _from_dict_v1(data_dict, base_path=None):
        if "endpoint" in data_dict:
            s3_dict = data_dict
            s3_dict["uri"] = data_dict["mount_absolute_path"]
            return VerifiableDatasetInfo(
                [S3DataSource.from_dict(ds_dict=s3_dict)],
                label=data_dict["label"],
                metadata=data_dict["metadata"],
                root_hash=data_dict["dataset_id"],
            )
        else:
            return VerifiableDatasetInfo(
                [LocalDataSource(source_path=".", base_path=base_path)],
                label=data_dict["label"],
                metadata=data_dict["metadata"],
                root_hash=data_dict["dataset_id"],
            )

    @staticmethod
    def _from_dict_v2(data_dict, base_path=None):
        """Deserialize the VerifiableDatasetInfo from JSON"""

        # Create appropriate data source based on dictionary information
        data_sources = []
        for datasource in data_dict["data_sources"]:
            if datasource["type"] == DataSourceType.LOCAL.value:
                data_source = LocalDataSource.from_dict(ds_dict=datasource, base_path=base_path)
            elif datasource["type"] == DataSourceType.S3.value:
                data_source = S3DataSource.from_dict(ds_dict=datasource)
            else:
                raise ValueError(f"Unknown storage type: {datasource['type']}")
            data_sources.append(data_source)

        return VerifiableDatasetInfo(
            data_sources,
            label=data_dict["label"],
            metadata=data_dict["metadata"],
            root_hash=data_dict["root_hash"],
        )

    @staticmethod
    def _from_dict(data_dict, base_path=None):
        """Deserialize the VerifiableDatasetInfo from JSON"""
        if "version" in data_dict and data_dict["version"] == "2.0":
            return VerifiableDatasetInfo._from_dict_v2(data_dict, base_path)
        return VerifiableDatasetInfo._from_dict_v1(data_dict, base_path)

    @staticmethod
    def from_json(json_str, base_path=None):
        """Deserialize the VerifiableDatasetInfo from JSON"""
        data_dict = json.loads(json_str)
        return VerifiableDatasetInfo._from_dict(data_dict, base_path)
