import json
from openfl.utilities.verifdataset.local_data_source import LocalDataSource
from openfl.utilities.verifdataset.verifiable_dataset_info import DatasetFormat, VerifiableDatasetInfo
import pytest
import os

from pathlib import Path
from typing import List, Tuple

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

###### Concise tests ######
def test_one_local_datasource_concise(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_concise(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_one_local_datasource_one_folder_concise(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_one_local_datasource_one_file_concise(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_two_dirs_concise(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1", ds2 / "1"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_two_files_concise(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file2.txt", ds2 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_one_local_datasource_two_files_concise(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file1.txt", ds1 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_different_base_path_concise(fs, data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    # Copy the datasources to a new location to have a different base_path
    new_base = Path("/new_test_data")
    new_ds1 = new_base / "datasource1"
    new_ds2 = new_base / "datasource2"
    copy_subtree(fs, ds1, new_ds1)
    copy_subtree(fs, ds2, new_ds2)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, str), f"Expected str, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifiable_json, new_base)

def test_two_local_datasource_use_saved_hash_concise(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    dataset_info = json.loads(verifaible_json)
    assert verifiable.verify_dataset(dataset_info)
    assert verifiable.verify_dataset()

def test_one_local_datasource_concise_verify_single_file(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, str), f"Expected str, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    verifiable1 = VerifiableDatasetInfo.from_dict(json.loads(verifaible_json), base_path)
    with pytest.raises(ValueError, match="verify_single_file is only valid for verbose datasets"):
        verifiable1.verify_single_file("dummy/file", hash)

def test_two_local_datasource_concise_with_symlink(fs, data_sources):
    real_ds1, ds2 = data_sources
    symlink_ds1 = Path("/symlink_datasource1")
    fs.create_symlink(symlink_ds1, real_ds1)  # Create symlink to real_ds1
    base_path, relative_paths = split_to_base_and_relative_paths([symlink_ds1, ds2])
    assert relative_paths[0] == real_ds1.name
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, str), f"Expected str, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifiable_json, base_path)

def test_two_local_datasource_different_base_path_concise_with_symlink(fs, data_sources):
    real_ds1, ds2 = data_sources
    symlink_ds1 = Path("/symlink_datasource1")
    fs.create_symlink(symlink_ds1, real_ds1)  # Create symlink to real_ds1
    base_path, relative_paths = split_to_base_and_relative_paths([symlink_ds1, ds2])
    assert relative_paths[0] == real_ds1.name
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.CONCISE, base_path=base_path)
    # Copy the datasources to a new location to have a different base_path
    new_base = Path("/new_test_data")
    new_ds1 = new_base / "datasource1"
    new_ds2 = new_base / "datasource2"
    copy_subtree(fs, symlink_ds1, new_ds1)
    copy_subtree(fs, ds2, new_ds2)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, str), f"Expected str, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifiable_json, new_base)

###### Verbose tests ######
def test_one_local_datasource_verbose_verify_single_file(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hashes = verifiable.create_dataset_hash()
    assert isinstance(hashes, dict), f"Expected dict, got {type(hashes)}"
    verifaible_json = verifiable.to_json()
    verifiable1 = VerifiableDatasetInfo.from_dict(json.loads(verifaible_json), base_path)
    for file_path, hash in hashes.items():
        file_full_path = Path(base_path) / Path(file_path)
        assert verifiable1.verify_single_file(file_full_path, hash)

def test_one_local_datasource_verbose(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hashes = verifiable.create_dataset_hash()
    assert isinstance(hashes, dict), f"Expected dict, got {type(hashes)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_verbose(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, dict), f"Expected dict, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_one_local_datasource_one_folder_verbose(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, dict), f"Expected dict, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_one_local_datasource_one_file_verbose(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, dict), f"Expected dict, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_two_dirs_concise(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1", ds2 / "1"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, dict), f"Expected dict, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_two_files_verbose(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file2.txt", ds2 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, dict), f"Expected dict, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_one_local_datasource_two_files_verbose(data_sources):
    ds1, _ = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1 / "1" / "file1.txt", ds1 / "1" / "file2.txt"])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, dict), f"Expected dict, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, base_path)

def test_two_local_datasource_different_base_path_verbose(fs, data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    # Copy the datasources to new location to have different base_path
    new_base = Path("/new_test_data")
    new_ds1 = new_base / "datasource1"
    new_ds2 = new_base / "datasource2"
    copy_subtree(fs, ds1, new_ds1)
    copy_subtree(fs, ds2, new_ds2)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, dict), f"Expected dict, got {type(dataset_hash)}"
    verifaible_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifaible_json, new_base)

def test_two_local_datasource_use_saved_hash_verbose(data_sources):
    ds1, ds2 = data_sources
    base_path, relative_paths = split_to_base_and_relative_paths([ds1, ds2])
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    hash = verifiable.create_dataset_hash()
    assert isinstance(hash, dict), f"Expected dict, got {type(hash)}"
    verifaible_json = verifiable.to_json()
    dataset_info = json.loads(verifaible_json)
    assert verifiable.verify_dataset(dataset_info)
    assert verifiable.verify_dataset()

def test_two_local_datasource_verbose_with_symlink(fs, data_sources):
    real_ds1, ds2 = data_sources
    symlink_ds1 = Path("/symlink_datasource1")
    fs.create_symlink(symlink_ds1, real_ds1)  # Create symlink to real_ds1
    base_path, relative_paths = split_to_base_and_relative_paths([symlink_ds1, ds2])
    assert relative_paths[0] == real_ds1.name
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, dict), f"Expected dict, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifiable_json, base_path)

def test_two_local_datasource_different_base_path_verbose_with_symlink(fs, data_sources):
    real_ds1, ds2 = data_sources
    symlink_ds1 = Path("/symlink_datasource1")
    fs.create_symlink(symlink_ds1, real_ds1)  # Create symlink to real_ds1
    base_path, relative_paths = split_to_base_and_relative_paths([symlink_ds1, ds2])
    assert relative_paths[0] == real_ds1.name
    datasources = [LocalDataSource(source_path=rel_path) for rel_path in relative_paths]
    verifiable = VerifiableDatasetInfo(data_sources=datasources, label="my_dataset", metadata="md", dataset_format=DatasetFormat.VERBOSE, base_path=base_path)
    # Copy the datasources to a new location to have a different base_path
    new_base = Path("/new_test_data")
    new_ds1 = new_base / "datasource1"
    new_ds2 = new_base / "datasource2"
    copy_subtree(fs, symlink_ds1, new_ds1)
    copy_subtree(fs, ds2, new_ds2)
    dataset_hash = verifiable.create_dataset_hash()
    assert isinstance(dataset_hash, dict), f"Expected dict, got {type(dataset_hash)}"
    verifiable_json = verifiable.to_json()
    assert VerifiableDatasetInfo.deserialize_and_verify(verifiable_json, new_base)
