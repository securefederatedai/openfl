# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from hashlib import sha256, sha384
import json
from openfl.federated.data.sources.local_data_source import LocalDataSource
from openfl.federated.data.sources.s3_data_source import S3DataSource
from openfl.federated.data.sources.verifiable_dataset_info import VerifiableDatasetInfo
import pytest
import boto3
import os

from pathlib import Path
from typing import List, Tuple
from moto import mock_aws

@pytest.fixture
def local_data_sources(fs) -> Tuple[Path, Path]:
    """Fixture to create two data sources with a file tree structure using pyfakefs."""
    base_tmp = Path("/test_data")  # Fake base path

    # Define datasource paths
    ds1 = base_tmp / "datasource1"
    ds2 = base_tmp / "datasource2"

    for ds in [ds1, ds2]:
        fs.create_dir(ds / "1")
        fs.create_dir(ds / "2")

        for subdir in ["1", "2"]:
            for i in range(1, 4):
                file_path = ds / subdir / f"file{i}.txt"
                fs.create_file(file_path, contents=f"Hello world! {ds.name} dir {subdir} file{i}\n")

    return ds1, ds2  # Return fake paths

@pytest.fixture
def mock_s3_buckets():
    with mock_aws():
        s3_client = boto3.client("s3")

        # Create two test buckets
        bucket1 = "test-bucket-1"
        bucket2 = "test-bucket-2"
        s3_client.create_bucket(Bucket=bucket1)
        s3_client.create_bucket(Bucket=bucket2)

        # Add multiple test files to both buckets
        files1 = [
            "folder1/file1.txt",
            "folder1/file2.txt",
            "folder1/subfolder/file3.txt",
            "folder2/file4.txt",
            "folder2/file5.txt",
            "folder2/subfolder/file6.txt"
        ]
        files2 = [
            "dir1/fileA.txt",
            "dir1/fileB.txt",
            "dir1/subdir/fileC.txt",
            "dir2/fileD.txt",
            "dir2/fileE.txt",
            "dir2/subdir/fileF.txt"
        ]

        for file in files1:
            s3_client.put_object(Bucket=bucket1, Key=file, Body=f"Content of {file}")

        for file in files2:
            s3_client.put_object(Bucket=bucket2, Key=file, Body=f"Content of {file}")

        yield bucket1, bucket2  # Return both bucket names

def split_to_base_and_relative_paths(data_sources_paths: List[str]) -> Tuple[str, List[str]]:
    """Split a list of paths into a base directory and relative paths."""
    absolute_paths = [os.path.realpath(path) for path in data_sources_paths]
    base_path = os.path.commonpath(absolute_paths)
    relative_paths = [os.path.relpath(path, base_path) for path in absolute_paths]
    return base_path, relative_paths

def copy_subtree(fs, existing_dir_path, new_dir_tree):
    """Recursively copy all files and directories from existing_dir_path to new_dir_tree in pyfakefs."""
    fs.create_dir(new_dir_tree)  # Ensure the new directory exists

    for subpath in existing_dir_path.iterdir():
        new_path = new_dir_tree / subpath.name
        if subpath.is_dir():
            copy_subtree(fs, subpath, new_path)  # Recursively copy subdirectories
        else:
            file_content = fs.get_object(subpath).contents  # Read from fake filesystem
            fs.create_file(new_path, contents=file_content)  # Create file in new location

def test_one_local_datasource(local_data_sources):
    ds1, _ = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_two_local_datasource(local_data_sources):
    ds1, ds2 = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_one_local_datasource_one_folder(local_data_sources):
    ds1, _ = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1"])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_one_local_datasource_one_file(local_data_sources):
    ds1, _ = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_two_local_datasource_two_dirs(local_data_sources):
    ds1, ds2 = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1", ds2 / "1"])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_two_local_datasource_two_files(local_data_sources):
    ds1, ds2 = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file2.txt", ds2 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_one_local_datasource_two_files(local_data_sources):
    ds1, _ = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file1.txt", ds1 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_two_local_datasource_different_base_path(fs, local_data_sources):
    ds1, ds2 = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    # Copy the datasources to a new location to have a different base_path
    new_base = Path("/new_test_data")
    new_ds1 = new_base / "datasource1"
    new_ds2 = new_base / "datasource2"
    copy_subtree(fs, ds1, new_ds1)
    copy_subtree(fs, ds2, new_ds2)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, str), f"Expected str, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_two_local_datasource_use_saved_hash(local_data_sources):
    ds1, ds2 = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    dataset_info = json.loads(verifiable_json)
    with pytest.raises(Exception):
        verifiable.verify_dataset()
    assert verifiable.verify_dataset(dataset_info["root_hash"])
    assert verifiable.verify_dataset()

def test_two_local_datasource_with_symlink(fs, local_data_sources):
    real_ds1, ds2 = local_data_sources
    symlink_ds1 = Path("/symlink_datasource1")
    fs.create_symlink(symlink_ds1, real_ds1)  # Create symlink to real_ds1
    base_path, relative_paths = split_to_base_and_relative_paths([symlink_ds1, ds2])
    assert relative_paths[0] == real_ds1.name
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, str), f"Expected str, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_two_local_datasource_different_base_path_with_symlink(fs, local_data_sources):
    real_ds1, ds2 = local_data_sources
    symlink_ds1 = Path("/symlink_datasource1")
    fs.create_symlink(symlink_ds1, real_ds1)  # Create symlink to real_ds1
    base_path, relative_paths = split_to_base_and_relative_paths([symlink_ds1, ds2])
    assert relative_paths[0] == real_ds1.name
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    # Copy the datasources to a new location to have a different base_path
    new_base = Path("/new_test_data")
    new_ds1 = new_base / "datasource1"
    new_ds2 = new_base / "datasource2"
    copy_subtree(fs, symlink_ds1, new_ds1)
    copy_subtree(fs, ds2, new_ds2)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, str), f"Expected str, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()

def test_one_local_datasource_verify_single_file(local_data_sources):
    ds1, _ = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    dataset_info_dict = json.loads(verifiable_json)
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    # Create & save in memory the hashes for all files
    verifiable.verify_dataset(dataset_info_dict["dataset_id"])
    for file_path, hash in verifiable.all_hashes.items():
        assert verifiable_from_json.verify_single_file(file_path, hash)

def test_two_local_datasource_verify_single_file(local_data_sources):
    ds1, ds2 = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    dataset_info_dict = json.loads(verifiable_json)
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    # Create & save in memory the hashes for all files
    verifiable.verify_dataset(dataset_info_dict["root_hash"])
    for file_path, hash in verifiable.all_hashes.items():
        assert verifiable_from_json.verify_single_file(file_path, hash)

def test_two_local_datasource_non_defalt_args(local_data_sources):
    ds1, ds2 = local_data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path, hash_func=sha256, max_dataset_size=500) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json, base_path)
    assert verifiable_from_json.verify_dataset()
    assert verifiable_from_json.data_sources[0].hash_func == sha256
    assert verifiable_from_json.data_sources[0].max_dataset_size == 500
    assert verifiable_from_json.data_sources[1].hash_func == sha256
    assert verifiable_from_json.data_sources[1].max_dataset_size == 500

def test_one_s3_data_source(mock_s3_buckets):
    bucket1, _ = mock_s3_buckets
    ds1 = S3DataSource(f"s3://{bucket1}/folder1")
    verifiable = VerifiableDatasetInfo(data_sources=[ds1], label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json)
    assert verifiable_from_json.verify_dataset()

def test_two_s3_datasource(mock_s3_buckets):
    bucket1, bucket2 = mock_s3_buckets
    ds1 = S3DataSource(f"s3://{bucket1}/folder1")
    ds2 = S3DataSource(f"s3://{bucket2}/dir1")
    verifiable = VerifiableDatasetInfo(data_sources=[ds1, ds2], label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json)
    assert verifiable_from_json.verify_dataset()

def test_one_s3_datasource_one_file(mock_s3_buckets):
    bucket1, _ = mock_s3_buckets
    ds1 = S3DataSource(f"s3://{bucket1}/folder1/file1.txt")
    verifiable = VerifiableDatasetInfo(data_sources=[ds1], label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json)
    assert verifiable_from_json.verify_dataset()

def test_two_s3_datasource_use_saved_hash(mock_s3_buckets):
    bucket1, bucket2 = mock_s3_buckets
    ds1 = S3DataSource(f"s3://{bucket1}/folder1")
    ds2 = S3DataSource(f"s3://{bucket2}/dir1")
    verifiable = VerifiableDatasetInfo(data_sources=[ds1, ds2], label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    dataset_info = json.loads(verifiable_json)
    assert verifiable.verify_dataset(dataset_info["root_hash"])
    assert verifiable.verify_dataset()

def test_one_s3_datasource_one_file_hash_func(mock_s3_buckets):
    bucket1, _ = mock_s3_buckets
    ds1 = S3DataSource(f"s3://{bucket1}/folder1/file1.txt", hash_func=sha384)
    verifiable = VerifiableDatasetInfo(data_sources=[ds1], label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json)
    assert verifiable_from_json.verify_dataset()

def test_two_s3_datasource_hash_func_use_saved_hash(mock_s3_buckets):
    bucket1, bucket2 = mock_s3_buckets
    ds1 = S3DataSource(f"s3://{bucket1}/folder1", hash_func=sha384)
    ds2 = S3DataSource(f"s3://{bucket2}/dir1", hash_func=sha384)
    verifiable = VerifiableDatasetInfo(data_sources=[ds1, ds2], label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    dataset_info = json.loads(verifiable_json)
    assert verifiable.verify_dataset(dataset_info["root_hash"])
    assert verifiable.verify_dataset()

def test_one_s3_datasource_verify_single_file(mock_s3_buckets):
    bucket1, _ = mock_s3_buckets
    ds1 = S3DataSource(f"s3://{bucket1}/folder1")
    verifiable = VerifiableDatasetInfo(data_sources=[ds1], label="my_dataset", metadata="md")
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifiable_json = verifiable.to_json()
    dataset_info_dict = json.loads(verifiable_json)
    verifiable_from_json = VerifiableDatasetInfo.from_json(verifiable_json)
    # Create & save in memory the hashes for all files
    verifiable.verify_dataset(dataset_info_dict["dataset_id"])
    for file_path, hash in verifiable.all_hashes.items():
        assert verifiable_from_json.verify_single_file(file_path, hash)
