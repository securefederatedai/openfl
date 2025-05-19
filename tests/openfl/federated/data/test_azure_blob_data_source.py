# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pytest
from unittest.mock import MagicMock, patch
from openfl.federated.data.sources.azure_blob_data_source import AzureBlobDataSource

@pytest.fixture
def mocked_azure_blob_ds():
    connection_string = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fakekey;"
    container_name = "test-container"

    # File paths and their binary content
    files = [
        "folder1/file1.txt",
        "folder1/file2.txt",
        "folder2/subfolder/file3.txt",
        "folder2/subfolder/file4.txt",
        "folder3/file5.txt",
        "folder3/subfolder/file6.txt",
    ]
    file_contents = {f: f"file content of file #{i}".encode() for i, f in enumerate(files)}

    with patch("azure.storage.blob.BlobServiceClient") as mock_service_cls:
        # Mock container and blob service
        mock_service = MagicMock()
        mock_container = MagicMock()

        # Mock list_blobs to return blob-like objects with a `name` attribute
        blob_mocks = []
        for f in files:
            mock_blob = MagicMock()
            type(mock_blob).name = property(lambda self, name=f: name)  # ensures blob.name == f
            blob_mocks.append(mock_blob)

        mock_container.list_blobs.return_value = blob_mocks

        # Mock get_blob_client to return blob clients with readall data
        def get_blob_client(blob_path):
            mock_blob_client = MagicMock()
            downloader = MagicMock()
            downloader.readall.return_value = file_contents[blob_path]
            mock_blob_client.download_blob.return_value = downloader
            return mock_blob_client

        mock_container.get_blob_client.side_effect = get_blob_client

        def list_blobs(name_starts_with=None):
            if name_starts_with is None:
                return blob_mocks
            return [b for b in blob_mocks if b.name.startswith(name_starts_with)]

        mock_container.list_blobs.side_effect = list_blobs

        mock_service.get_container_client.return_value = mock_container
        mock_service_cls.from_connection_string.return_value = mock_service

        yield connection_string, container_name, files, file_contents

def test_enumerate_files(mocked_azure_blob_ds):
    connection_string, container_name, expected_files, _ = mocked_azure_blob_ds
    datasource = AzureBlobDataSource(
            name="abds",
            connection_string=connection_string,
            container_name=container_name,
        )
    listed_files = datasource.enumerate_files()
    assert listed_files == expected_files

def test_compute_file_hash(mocked_azure_blob_ds):
    connection_string, container_name, files, file_contents = mocked_azure_blob_ds
    datasource = AzureBlobDataSource(
            name="abds",
            connection_string=connection_string,
            container_name=container_name,
        )
    for f in files:
        expected_hash = hashlib.sha384(file_contents[f]).hexdigest()
        actual_hash = datasource.compute_file_hash(f)
        assert actual_hash == expected_hash, f"Hash mismatch for file: {f}"

def test_from_dict(mocked_azure_blob_ds):
    connection_string, container_name, _, _ = mocked_azure_blob_ds

    ds_dict = {
        "name": "abds",
        "connection_string": connection_string,
        "container_name": container_name,
        "hash_func": "sha384"
    }

    new_ds = AzureBlobDataSource.from_dict(ds_dict)
    assert new_ds.connection_string == connection_string
    assert new_ds.container_name == container_name
    assert callable(new_ds.hash_func)
    assert new_ds.hash_func == hashlib.sha384

def test_files_content(mocked_azure_blob_ds):
    connection_string, container_name, files, file_contents = mocked_azure_blob_ds
    datasource = AzureBlobDataSource(
            name="abds",
            connection_string=connection_string,
            container_name=container_name,
        )
    for f in files:
        content = datasource.read_blob(f)
        assert content == file_contents[f], f"Content mismatch for file: {f}"

def test_folder_prefix(mocked_azure_blob_ds):
    connection_string, container_name, files, _ = mocked_azure_blob_ds
    folder_prefix = "folder1/"
    datasource = AzureBlobDataSource(
            name="abds",
            connection_string=connection_string,
            container_name=container_name,
            folder_prefix=folder_prefix,
        )
    expected_files = [f for f in files if f.startswith(folder_prefix)]
    listed_files = datasource.enumerate_files()
    assert listed_files == expected_files
