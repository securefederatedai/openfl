# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

from pathlib import Path
from typing import List, Tuple
import pytest

from openfl.federated.data.sources.data_source import DataSourceType
from openfl.federated.data.sources.local_data_source import LocalDataSource
from openfl.federated.data.sources.torch.local_folder import LabelMapper, LocalFolder
from openfl.federated.data.sources.torch.verifiable_map_style_dataset import VerifiableMapStyleDataset

from openfl.federated.data.sources.torch.verifiable_map_style_image_folder import VerifiableImageFolder
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

class LocalTextFolder(LocalFolder):
    """Custom dataset that loads all images from a local directory tree and assigns labels based on folder names."""
    def load_file(self, file_path):
        """Load a file from the dataset."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

class MockVerifiableMapStyle(VerifiableMapStyleDataset):

    def __init__(self, vds, transform=None, verify_dataset_items=False):
        self.label_mapper = LabelMapper()
        super().__init__(vds, transform=transform, verify_dataset_items=verify_dataset_items)

    def create_datasets(self):
        datasources = []
        for data_source in self.verifiable_dataset_info.data_sources:
            if data_source.type == DataSourceType.LOCAL:
                datasources.append(LocalTextFolder(base_path=data_source.get_source_full_path(), label_mapper=self.label_mapper, transform=self.transform))
            else:
                raise ValueError(f"Unknown or unsupported storage type: {data_source.type}")
        return datasources


def test_local_map_style_datasource_verbose(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
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

def test_local_map_style_datasource_verbose_verify(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
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


def test_local_image_folder_map_style_datasource_verbose(fake_image_datasources):
    ds1, ds2, _ = fake_image_datasources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
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

def test_local_image_folder_map_style_datasource_verbose_verify(fake_image_datasources):
    ds1, ds2, _ = fake_image_datasources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
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


def test_local_image_folder_map_style_datasource_verbose_labels(fake_image_datasources):
    """Test that LabelMapper correctly maps labels across multiple datasets."""
    ds1, ds2, ds3 = fake_image_datasources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2, ds3])
    datasources = [LocalDataSource(source_path=rel_path, base_path=base_path) for rel_path in relative_paths]
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
