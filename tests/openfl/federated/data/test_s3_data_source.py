# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import boto3
from moto import mock_aws
from hashlib import sha384
from openfl.federated.data.sources.s3_data_source import S3DataSource


@pytest.fixture
def mock_s3_bucket():
    """Mock an S3 bucket with a structured file hierarchy."""
    with mock_aws():
        s3_client = boto3.client("s3")
        bucket_name = "test-bucket"
        s3_client.create_bucket(Bucket=bucket_name)

        # Add test files
        files = [
            "folder1/file1.txt",
            "folder1/file2.txt",
            "folder2/subfolder/file3.txt",
            "folder2/subfolder/file4.txt",
            "folder3/file5.txt",
            "folder3/subfolder/file6.txt",
        ]
        for file in files:
            s3_client.put_object(Bucket=bucket_name, Key=file, Body=f"Data for {file}".encode())

        yield bucket_name


@pytest.fixture
def s3_data_source(mock_s3_bucket):
    """Creates an S3DataSource instance for the test bucket."""
    return S3DataSource("s3ds", f"s3://{mock_s3_bucket}/")

def test_enumerate_files(s3_data_source, mock_s3_bucket):
    """Test that enumerate_files returns full S3 URIs."""
    expected_files = [
        f"s3://{mock_s3_bucket}/folder1/file1.txt",
        f"s3://{mock_s3_bucket}/folder1/file2.txt",
        f"s3://{mock_s3_bucket}/folder2/subfolder/file3.txt",
        f"s3://{mock_s3_bucket}/folder2/subfolder/file4.txt",
        f"s3://{mock_s3_bucket}/folder3/file5.txt",
        f"s3://{mock_s3_bucket}/folder3/subfolder/file6.txt",
    ]

    enumerated_files = list(s3_data_source.enumerate_files())
    assert set(enumerated_files) == set(expected_files), \
        f"Expected {expected_files}, but got {enumerated_files}"


def test_get_s3_etag(s3_data_source, mock_s3_bucket):
    """Test that _get_s3_etag retrieves the correct ETag."""
    file_path = f"s3://{mock_s3_bucket}/folder1/file1.txt"
    etag = s3_data_source._get_s3_etag(file_path)
    assert etag is not None, "ETag should not be None"
    assert isinstance(etag, str), "ETag should be a string"


def test_read_s3_object(s3_data_source, mock_s3_bucket):
    """Test that _read_s3_object correctly retrieves file content."""
    file_path = f"s3://{mock_s3_bucket}/folder1/file1.txt"
    content = s3_data_source._read_s3_object(file_path)
    expected_content = b"Data for folder1/file1.txt"

    assert content == expected_content, \
        f"Expected '{expected_content}', but got '{content}'"


def test_compute_file_hash_default(s3_data_source, mock_s3_bucket):
    """Test compute_file_hash without a custom hash function (should return ETag)."""
    file_path = f"s3://{mock_s3_bucket}/folder1/file1.txt"
    expected_etag = s3_data_source._get_s3_etag(file_path)

    computed_hash = s3_data_source.compute_file_hash(file_path)

    assert computed_hash == expected_etag, \
        f"Expected ETag '{expected_etag}', but got '{computed_hash}'"


@pytest.fixture
def s3_data_source_with_hash(mock_s3_bucket):
    """Creates an S3DataSource instance with a custom hash function."""
    return S3DataSource("s3ds", f"s3://{mock_s3_bucket}/", hash_func=sha384)


def test_compute_file_hash_with_function(s3_data_source_with_hash, mock_s3_bucket):
    """Test compute_file_hash with a custom hash function."""
    file_path = f"s3://{mock_s3_bucket}/folder1/file1.txt"
    file_content = b"Data for folder1/file1.txt"
    expected_hash = sha384(file_content).hexdigest()

    computed_hash = s3_data_source_with_hash.compute_file_hash(file_path)

    assert computed_hash == expected_hash, \
        f"Expected hash '{expected_hash}', but got '{computed_hash}'"
