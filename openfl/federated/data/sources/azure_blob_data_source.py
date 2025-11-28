# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
from importlib import util
from typing import Callable

from openfl.federated.data.sources.data_source import DataSource, DataSourceType

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)


class AzureBlobDataSource(DataSource):
    """Class for Azure Blob data"""

    def __init__(
        self,
        name: str,
        connection_string: str,
        container_name: str,
        folder_prefix="",
        hash_func: Callable[..., "hashlib._Hash"] = hashlib.sha384,
    ):
        # Check if azure.storage.blob is installed.
        if util.find_spec("azure.storage.blob") is None:
            raise Exception(
                "'azure-storage-blob' not installed."
                "This package is necessary for interacting with Azure Blob Storage."
            )
        from azure.storage.blob import BlobServiceClient

        super().__init__(DataSourceType.AZURE_BLOB, name)
        self.connection_string = connection_string
        self.container_name = container_name
        self.folder_prefix = folder_prefix
        if not super().is_valid_hash_function(hash_func):
            raise ValueError(
                f"Data source {self.name}: Invalid hash function: {hash_func.__name__}."
                " Must be a hashlib function."
            )
        self.hash_func = hash_func
        self._service_client = BlobServiceClient.from_connection_string(connection_string)
        self._container_client = self._service_client.get_container_client(container_name)

    def enumerate_files(self):
        """List all blobs in the container (full recursive listing)"""
        return [
            blob.name
            for blob in self._container_client.list_blobs(name_starts_with=self.folder_prefix)
        ]

    def compute_file_hash(self, blob_path: str):
        """Compute the hash of a blob's binary content."""
        content = self.read_blob(blob_path)
        return self.hash_func(content).hexdigest()

    def read_blob(self, blob_path: str):
        """Read blob content as binary data."""
        blob_client = self._container_client.get_blob_client(blob_path)
        downloader = blob_client.download_blob()
        return downloader.readall()

    @classmethod
    def from_dict(cls, ds_dict: dict):
        hash_func = getattr(hashlib, ds_dict.get("hash_func", "sha384"), None)
        return cls(
            name=ds_dict["name"],
            connection_string=ds_dict["connection_string"],
            container_name=ds_dict["container_name"],
            hash_func=hash_func,
        )
