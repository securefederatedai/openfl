# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch
import pytest
import boto3
from moto import mock_aws
import torch
from torchvision.transforms import ToTensor

import io
from openfl.federated.data.sources.azure_blob_data_source import AzureBlobDataSource
from openfl.federated.data.sources.data_source import DataSource
from openfl.federated.data.sources.local_data_source import LocalDataSource
from openfl.federated.data.sources.torch.folder_dataset import FolderDataset, LabelMapper
from openfl.federated.data.sources.torch.verifiable_map_style_dataset import VerifiableMapStyleDataset

from openfl.federated.data.sources.torch.verifiable_map_style_image_folder import VerifiableImageFolder
from openfl.federated.data.sources.s3_data_source import S3DataSource
from openfl.federated.data.sources.verifiable_dataset_info import VerifiableDatasetInfo

from PIL import Image

@pytest.fixture
def data_sources(fs) -> Tuple[Path, Path]:
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

def split_to_base_and_relative_paths(data_sources_paths: List[str]) -> Tuple[str, List[str]]:
    """Split a list of paths into a base directory and relative paths."""
    absolute_paths = [os.path.realpath(path) for path in data_sources_paths]
    base_path = os.path.commonpath(absolute_paths)
    relative_paths = [os.path.relpath(path, base_path) for path in absolute_paths]
    return base_path, relative_paths

class TextFolder(FolderDataset):

    def __init__(self, datasource: DataSource, label_mapper: LabelMapper, transform=None):
        """
        Args:
            datasource (DataSource): DataSource object representing the data source.
            label_mapper (LabelMapper): LabelMapper object to map class names to indices.
            transform (callable, optional): Transformations to apply to loaded data.
        """
        self.datasource = datasource
        super().__init__(label_mapper, transform=transform)

    def load_file(self, file_path):
        """Load a file from the dataset."""
        return self.datasource.read_blob(file_path).decode("utf-8")

class MockVerifiableMapStyle(VerifiableMapStyleDataset):

    def __init__(self, vds, transform=None, verify_dataset_items=False):
        self.label_mapper = LabelMapper()
        super().__init__(vds, transform=transform, verify_dataset_items=verify_dataset_items)

    def create_datasets(self):
        datasources = []
        for data_source in self.verifiable_dataset_info.data_sources:
            datasources.append(TextFolder(data_source, label_mapper=self.label_mapper, transform=self.transform))
        return datasources


def test_local_map_style_datasource(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(name="lds", source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = MockVerifiableMapStyle(verifiable_dataset_info, verify_dataset_items=False)

    assert len(verifiable_map_style) == 12

    # Check files contents
    for i in range(len(verifiable_map_style)):
        assert verifiable_map_style[i][0] == f"Hello world! datasource{i//6 + 1} dir {i//3 % 2 + 1} file{i % 3 + 1}\n"

def test_local_map_style_datasource_verify(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(name="lds", source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    dataset_info_json = verifiable_dataset_info.to_json()
    verifiable_dataset_info.verify_dataset(json.loads(dataset_info_json)["root_hash"])
    verifiable_dataset_info.verify_dataset()
    verifiable_map_style = MockVerifiableMapStyle(verifiable_dataset_info, verify_dataset_items=True)
    assert len(verifiable_map_style) == 12

    # Check files contents
    for i in range(len(verifiable_map_style)):
        assert verifiable_map_style[i][0] == f"Hello world! datasource{i//6 + 1} dir {i//3 % 2 + 1} file{i % 3 + 1}\n"


@pytest.fixture
def fake_image_datasources(fs):
    """Fixture to create two fake image datasets using pyfakefs."""
    base_path_1 = Path("/fake/dataset1")
    base_path_2 = Path("/fake/dataset2")
    base_path_3 = Path("/fake/dataset3")

    # Dataset 1 and 2 have identical labels
    labels_ds1_ds2 = ["cat", "dog"]
    # Dataset 3 introduces new labels but retains one ("dog")
    labels_ds3 = ["dog", "rabbit", "elephant"]
    datasets = [
            (base_path_1, labels_ds1_ds2),
            (base_path_2, labels_ds1_ds2),
            (base_path_3, labels_ds3),
        ]

    # Create both dataset directories with labeled images
    for base_path, labels in datasets:
        for label in labels:
            label_path = base_path / label
            fs.create_dir(str(label_path))
            for i in range(3):  # Create 3 images per label
                img_path = label_path / f"image_{i}.png"
                img = Image.new('RGB', (100, 100), color='red')
                img.save(str(img_path), format='PNG')

    return base_path_1, base_path_2, base_path_3  # Provide dataset paths to tests


def test_local_image_folder_map_style_datasource(fake_image_datasources):
    ds1, ds2, _ = fake_image_datasources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(name="lds", source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=False)
    assert len(verifiable_map_style) == 12
    assert len(verifiable_map_style.datasets) == len(datasources)

    for i in range(len(verifiable_map_style)):
        if i < 6:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[0][i]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[0][i]["label"]
        else:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[1][i - 6]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[1][i - 6]["label"]

def test_local_image_folder_map_style_datasource_verify(fake_image_datasources):
    ds1, ds2, _ = fake_image_datasources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(name="lds", source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    dataset_info_json = verifiable_dataset_info.to_json()
    verifiable_dataset_info.verify_dataset(json.loads(dataset_info_json)["root_hash"])
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=True)
    assert len(verifiable_map_style) == 12
    assert len(verifiable_map_style.datasets) == len(datasources)

    for i in range(len(verifiable_map_style)):
        if i < 6:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[0][i]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[0][i]["label"]
        else:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[1][i - 6]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[1][i - 6]["label"]


def test_local_image_folder_map_style_datasource_labels(fake_image_datasources):
    """Test that LabelMapper correctly maps labels across multiple datasets."""
    ds1, ds2, ds3 = fake_image_datasources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2, ds3])
    datasources = [LocalDataSource(name="lds", source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=True)

    # Check the label mapping
    label_mapper = verifiable_map_style.label_mapper
    assert label_mapper is not None, "LabelMapper should be initialized"

    # Expected label mappings (combining all dataset labels)
    expected_labels = {"cat", "dog", "rabbit", "elephant"}
    mapped_labels = set(label_mapper.label_to_idx.keys())

    assert mapped_labels == expected_labels, f"Expected labels {expected_labels}, but got {mapped_labels}"

    # Ensure all mapped values are unique integers
    mapped_values = set(label_mapper.label_to_idx.values())
    assert len(mapped_values) == len(expected_labels), "LabelMapper should assign unique integers to each label"

    # Ensure "dog" has the same integer mapping across all datasets
    dog_label_id = label_mapper.get_label_index("dog")

    # Ensure that index also maps back correctly in idx_to_label
    assert label_mapper.idx_to_label[dog_label_id] == "dog", "Reverse mapping is incorrect"

    # Esure that "dog" is mapped to a single label index
    dog_occurrences = sum(1 for label in label_mapper.idx_to_label.values() if label == "dog")
    assert dog_occurrences == 1, f"Label 'dog' should have only one unique index, but found {dog_occurrences}"

@pytest.fixture
def mock_s3_image_buckets():
    """Creates two mock S3 buckets with labeled image folders."""
    with mock_aws():
        s3_client = boto3.client("s3")
        bucket1_name = "test-image-bucket-1"
        bucket2_name = "test-image-bucket-2"

        s3_client.create_bucket(Bucket=bucket1_name)
        s3_client.create_bucket(Bucket=bucket2_name)

        # Function to generate a valid PNG image
        def generate_image():
            img = Image.new("RGB", (10, 10), color=(255, 0, 0))  # Create a red image
            img_bytes_io = io.BytesIO()
            img.save(img_bytes_io, format="PNG")
            return img_bytes_io.getvalue()  # Get binary data

        image_data = generate_image()  # Generate once and reuse

        # Define image files for each bucket
        bucket1_images = [
            ("train/cat/cat1.png", image_data),
            ("train/cat/cat2.png", image_data),
            ("train/dog/dog1.png", image_data),
            ("train/dog/dog2.png", image_data),
            ("train/rabbit/rabbit1.png", image_data),
            ("train/rabbit/rabbit2.png", image_data),
        ]

        bucket2_images = [
            ("rabbit/rabbit3.png", image_data),
            ("rabbit/rabbit4.png", image_data),
            ("elephant/elephant1.png", image_data),
            ("elephant/elephant2.png", image_data),
            ("snake/snake1.png", image_data),
            ("snake/snake2.png", image_data),
        ]

        # Upload images to the first bucket
        for key, data in bucket1_images:
            s3_client.put_object(Bucket=bucket1_name, Key=key, Body=data)

        # Upload images to the second bucket
        for key, data in bucket2_images:
            s3_client.put_object(Bucket=bucket2_name, Key=key, Body=data)

        yield f"s3://{bucket1_name}/train/", f"s3://{bucket2_name}/"


def test_s3_image_folder_map_style_datasource(mock_s3_image_buckets):
    ds1, ds2 = mock_s3_image_buckets
    datasources = [S3DataSource(name="s3ds", uri=ds1), S3DataSource(name="s3ds", uri=ds2)]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=False)
    assert len(verifiable_map_style) == 12
    assert len(verifiable_map_style.datasets) == len(datasources)

    for i in range(len(verifiable_map_style)):
        if i < 6:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[0][i]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[0][i]["label"]
        else:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[1][i - 6]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[1][i - 6]["label"]

def test_s3_image_folder_map_style_datasource_verify(mock_s3_image_buckets):
    ds1, ds2 = mock_s3_image_buckets
    datasources = [S3DataSource(name="s3ds", uri=ds1), S3DataSource(name="s3ds", uri=ds2)]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    dataset_info_json = verifiable_dataset_info.to_json()
    verifiable_dataset_info.verify_dataset(json.loads(dataset_info_json)["root_hash"])
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=True)
    assert len(verifiable_map_style) == 12
    assert len(verifiable_map_style.datasets) == len(datasources)

    for i in range(len(verifiable_map_style)):
        if i < 6:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[0][i]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[0][i]["label"]
        else:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[1][i - 6]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[1][i - 6]["label"]

def test_s3_image_folder_map_style_datasource_verify_w_transform(mock_s3_image_buckets):
    ds1, ds2 = mock_s3_image_buckets
    datasources = [S3DataSource(name="s3ds", uri=ds1), S3DataSource(name="s3ds", uri=ds2)]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    dataset_info_json = verifiable_dataset_info.to_json()
    verifiable_dataset_info.verify_dataset(json.loads(dataset_info_json)["root_hash"])
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, transform=ToTensor(), verify_dataset_items=True)
    assert len(verifiable_map_style) == 12
    assert len(verifiable_map_style.datasets) == len(datasources)

    for i in range(len(verifiable_map_style)):
        if i < 6:
            assert torch.equal(verifiable_map_style[i][0], verifiable_map_style.datasets[0][i]["data"])
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[0][i]["label"]
        else:
            assert torch.equal(verifiable_map_style[i][0], verifiable_map_style.datasets[1][i - 6]["data"])
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[1][i - 6]["label"]

def test_s3_image_folder_map_style_datasource_labels(mock_s3_image_buckets):
    """Test that LabelMapper correctly maps labels across multiple datasets."""
    s3_ds1, s3_ds2 = mock_s3_image_buckets
    datasources =[S3DataSource(name="s3ds", uri=s3_ds1), S3DataSource(name="s3ds", uri=s3_ds2)]

    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=True)

    # Check the label mapping
    label_mapper = verifiable_map_style.label_mapper
    assert label_mapper is not None, "LabelMapper should be initialized"

    # Expected label mappings (combining all dataset labels)
    expected_labels = {"cat", "dog", "rabbit", "elephant", "snake"}
    mapped_labels = set(label_mapper.label_to_idx.keys())

    assert mapped_labels == expected_labels, f"Expected labels {expected_labels}, but got {mapped_labels}"

    # Ensure all mapped values are unique integers
    mapped_values = set(label_mapper.label_to_idx.values())
    assert len(mapped_values) == len(expected_labels), "LabelMapper should assign unique integers to each label"

    # Ensure "elephant" has the same integer mapping across all datasets
    elephant_label_id = label_mapper.get_label_index("elephant")

    # Ensure that index also maps back correctly in idx_to_label
    assert label_mapper.idx_to_label[elephant_label_id] == "elephant", "Reverse mapping is incorrect"

    # Esure that "dog" is mapped to a single label index
    dog_occurrences = sum(1 for label in label_mapper.idx_to_label.values() if label == "dog")
    assert dog_occurrences == 1, f"Label 'dog' should have only one unique index, but found {dog_occurrences}"





@pytest.fixture
def mock_azure_blob():
    # Define two containers
    containers = [
        {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=fake1;AccountKey=fakekey1;",
            "container_name": "test-container-1",
        },
        {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=fake2;AccountKey=fakekey2;",
            "container_name": "test-container-2",
        }
    ]

    files1 = [
        "train/cat/cat1.png",
        "train/cat/cat2.png",
        "train/dog/dog1.png",
        "train/dog/dog2.png",
        "train/rabbit/rabbit1.png",
        "train/rabbit/rabbit2.png"
    ]
    files2 = [
        "rabbit/rabbit3.png",
        "rabbit/rabbit4.png",
        "elephant/elephant1.png",
        "elephant/elephant2.png",
        "snake/snake1.png",
        "snake/snake2.png"
    ]
    files_per_container = [files1, files2]

    img = Image.new("RGB", (10, 10), color="red")
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format="PNG")
    img_data = img_bytes_io.getvalue()

    file_contents_list = [
        {f: img_data for f in files}
        for files in files_per_container
    ]
    with patch("azure.storage.blob.BlobServiceClient") as mock_service_cls:
        service_mocks = []

        for files, file_contents in zip(files_per_container, file_contents_list):
            mock_service = MagicMock()
            mock_container = MagicMock()

            # Create blob mocks with correct names
            blob_mocks = []
            for f in files:
                mock_blob = MagicMock()
                type(mock_blob).name = property(lambda self, name=f: name)
                blob_mocks.append(mock_blob)
            mock_container.list_blobs.return_value = blob_mocks

            def get_blob_client(blob_path, file_contents=file_contents):
                mock_blob_client = MagicMock()
                downloader = MagicMock()
                downloader.readall.return_value = file_contents[blob_path]
                mock_blob_client.download_blob.return_value = downloader
                return mock_blob_client

            mock_container.get_blob_client.side_effect = get_blob_client
            mock_service.get_container_client.return_value = mock_container
            service_mocks.append(mock_service)

        def from_connection_string_mock(conn_str):
            if conn_str == containers[0]["connection_string"]:
                return service_mocks[0]
            elif conn_str == containers[1]["connection_string"]:
                return service_mocks[1]
            else:
                raise ValueError(f"Unknown connection string: {conn_str}")

        mock_service_cls.from_connection_string.side_effect = from_connection_string_mock

        yield containers[0], containers[1]



def test_azure_blob_image_folder_map_style_datasource(mock_azure_blob):
    container1, container2 = mock_azure_blob
    ds1 = AzureBlobDataSource("abds", container1["connection_string"], container1["container_name"])
    ds2 = AzureBlobDataSource("abds", container2["connection_string"], container2["container_name"])
    datasources = [ds1, ds2]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=False)
    assert len(verifiable_map_style) == 12
    assert len(verifiable_map_style.datasets) == len(datasources)

    for i in range(len(verifiable_map_style)):
        if i < 6:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[0][i]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[0][i]["label"]
        else:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[1][i - 6]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[1][i - 6]["label"]

def test_azure_blob_image_folder_map_style_datasource_verify(mock_azure_blob):
    container1, container2 = mock_azure_blob
    ds1 = AzureBlobDataSource("abds", container1["connection_string"], container1["container_name"])
    ds2 = AzureBlobDataSource("abds", container2["connection_string"], container2["container_name"])
    datasources = [ds1, ds2]
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    dataset_info_json = verifiable_dataset_info.to_json()
    verifiable_dataset_info.verify_dataset(json.loads(dataset_info_json)["root_hash"])
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=True)
    assert len(verifiable_map_style) == 12
    assert len(verifiable_map_style.datasets) == len(datasources)

    for i in range(len(verifiable_map_style)):
        if i < 6:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[0][i]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[0][i]["label"]
        else:
            assert verifiable_map_style[i][0] == verifiable_map_style.datasets[1][i - 6]["data"]
            assert verifiable_map_style[i][1] == verifiable_map_style.datasets[1][i - 6]["label"]


def test_azure_blob_image_folder_map_style_datasource_labels(mock_azure_blob):
    """Test that LabelMapper correctly maps labels across multiple datasets."""
    container1, container2 = mock_azure_blob
    ds1 = AzureBlobDataSource("abds", container1["connection_string"], container1["container_name"])
    ds2 = AzureBlobDataSource("abds", container2["connection_string"], container2["container_name"])
    datasources = [ds1, ds2]

    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=True)

    # Check the label mapping
    label_mapper = verifiable_map_style.label_mapper
    assert label_mapper is not None, "LabelMapper should be initialized"

    # Expected label mappings (combining all dataset labels)
    expected_labels = {"cat", "dog", "rabbit", "elephant", "snake"}
    mapped_labels = set(label_mapper.label_to_idx.keys())

    assert mapped_labels == expected_labels, f"Expected labels {expected_labels}, but got {mapped_labels}"

    # Ensure all mapped values are unique integers
    mapped_values = set(label_mapper.label_to_idx.values())
    assert len(mapped_values) == len(expected_labels), "LabelMapper should assign unique integers to each label"

    # Ensure "elephant" has the same integer mapping across all datasets
    elephant_label_id = label_mapper.get_label_index("elephant")

    # Ensure that index also maps back correctly in idx_to_label
    assert label_mapper.idx_to_label[elephant_label_id] == "elephant", "Reverse mapping is incorrect"

    # Esure that "dog" is mapped to a single label index
    dog_occurrences = sum(1 for label in label_mapper.idx_to_label.values() if label == "dog")
    assert dog_occurrences == 1, f"Label 'dog' should have only one unique index, but found {dog_occurrences}"


def test_mixed_ds_image_folder_map_style_datasource_labels(fake_image_datasources, mock_azure_blob):
    """Test that LabelMapper correctly maps labels across multiple datasets."""
    local_ds1, local_ds2, _ = fake_image_datasources
    base_path, relative_paths = split_to_base_and_relative_paths([local_ds1, local_ds2])
    datasources = [LocalDataSource(name="lds", source_path=rel_path, base_path=base_path) for rel_path in relative_paths]

    container1, container2 = mock_azure_blob
    azure_ds1 = AzureBlobDataSource("abds", container1["connection_string"], container1["container_name"])
    azure_ds2 = AzureBlobDataSource("abds", container2["connection_string"], container2["container_name"])

    datasources.append(azure_ds1)
    datasources.append(azure_ds2)
    verifiable_dataset_info = VerifiableDatasetInfo(
        data_sources=datasources,
        label="Test VerifiableMapStyleDataset",
        metadata={"test": "test"}
    )
    verifiable_map_style = VerifiableImageFolder(verifiable_dataset_info, verify_dataset_items=True)

    # Check the label mapping
    label_mapper = verifiable_map_style.label_mapper
    assert label_mapper is not None, "LabelMapper should be initialized"

    # Expected label mappings (combining all dataset labels)
    expected_labels = {"cat", "dog", "rabbit", "elephant", "snake"}
    mapped_labels = set(label_mapper.label_to_idx.keys())

    assert mapped_labels == expected_labels, f"Expected labels {expected_labels}, but got {mapped_labels}"

    # Ensure all mapped values are unique integers
    mapped_values = set(label_mapper.label_to_idx.values())
    assert len(mapped_values) == len(expected_labels), "LabelMapper should assign unique integers to each label"

    # Ensure "dog" has the same integer mapping across all datasets
    dog_label_id = label_mapper.get_label_index("dog")

    # Ensure that index also maps back correctly in idx_to_label
    assert label_mapper.idx_to_label[dog_label_id] == "dog", "Reverse mapping is incorrect"

    # Esure that "dog" is mapped to a single label index
    dog_occurrences = sum(1 for label in label_mapper.idx_to_label.values() if label == "dog")
    assert dog_occurrences == 1, f"Label 'dog' should have only one unique index, but found {dog_occurrences}"
