# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from glob import glob
import logging
import importlib
from pathlib import Path
import json

import tests.end_to_end.utils.defaults as defaults
import tests.end_to_end.utils.exceptions as ex
from tests.end_to_end.models import az_storage as az_storage_model ,s3_bucket as s3_model

log = logging.getLogger(__name__)


def setup_collaborator_data(collaborators, model_name, local_bind_path):
    """
    Function to setup the data for collaborators.
    IMP: This function is specific to the model and should be updated as per the model requirements.
    Args:
        collaborators (list): List of collaborator objects
        model_name (str): Model name
        local_bind_path (str): Local bind path
    """
    # Check if data already exists, if yes, skip the download part
    # This is mainly helpful in case of re-runs
    if all(os.path.exists(os.path.join(collaborator.workspace_path, "data", str(index))) for index, collaborator in enumerate(collaborators, start=1)):
        log.info("Data already exists for all the collaborators. Skipping the download part..")
        return
    else:
        log.info("Data does not exist for all the collaborators. Proceeding with the download..")
        # Below step will also modify the data.yaml file for all the collaborators
        if model_name == defaults.ModelName.XGB_HIGGS.value:
            download_higgs_data(collaborators, local_bind_path)
        elif model_name == defaults.ModelName.FLOWER_APP_PYTORCH.value:
            download_flower_data(collaborators, local_bind_path)

    log.info("Data setup is complete for all the collaborators")


def download_gandlf_data(aggregator, local_bind_path, num_collaborators, results_path):
    """
    Function to download the data for GanDLF segmentation test model and copy to the respective collaborator workspaces
    For GanDLF, data download happens at aggregator level, thus we can not call this function from setup_collaborator_data
    where download is at collaborator level
    Args:
        aggregator: Aggregator object
        collaborators: List of collaborator objects
        local_bind_path: Local bind path
        results_path: Result directory (mostly $HOME/results) where GaNDLF csv and config yaml files are present
    """
    try:
        # Get list of all CSV files in openfl_path
        csv_files = glob(os.path.join(results_path, '*.csv'))

        # Get data.yaml file and remove any entry, if present
        data_file = os.path.join(aggregator.workspace_path, "plan", "data.yaml")
        with open(data_file, "w") as df:
            df.write("")

        # Copy the data to the respective workspaces based on the index
        for col_index in range(1, num_collaborators+1):
            dst_folder = os.path.join(aggregator.workspace_path, "data", str(col_index))
            os.makedirs(dst_folder, exist_ok=True)
            for csv_file in csv_files:
                shutil.copy(csv_file, dst_folder)
                log.info(f"Copied data from {csv_file} to {dst_folder}")

            aggregator.modify_data_file(
                defaults.COL_DATA_FILE.format(local_bind_path, "aggregator"),
                f"collaborator{col_index}",
                col_index,
            )
    except Exception as e:
        raise ex.DataSetupException(f"Failed to modify the data file: {e}")

    return True


def copy_gandlf_data_to_collaborators(aggregator, collaborators, local_bind_path):
    """
    Function to copy the GaNDLF data from aggregator to respective collaborators
    """
    try:
        # Copy the data to the respective workspaces based on the index
        for index, collaborator in enumerate(collaborators, start=1):
            src_folder = os.path.join(aggregator.workspace_path, "data", str(index))
            dst_folder = os.path.join(collaborator.workspace_path, "data", str(index))
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                log.info(f"Copied data from {src_folder} to {dst_folder}")
            else:
                raise ex.DataSetupException(f"Source folder {src_folder} does not exist for {collaborator.name}")

            # Modify the data.yaml file for all the collaborators
            collaborator.modify_data_file(
                defaults.COL_DATA_FILE.format(local_bind_path, collaborator.name),
                index,
            )
    except Exception as e:
        raise ex.DataSetupException(f"Failed to modify the data file: {e}")


def download_flower_data(collaborators, local_bind_path):
    """
    Download the data for the model and copy to the respective collaborator workspaces
    Also modify the data.yaml file for all the collaborators
    Args:
        collaborators (list): List of collaborator objects
        local_bind_path (str): Local bind path
    Returns:
        bool: True if successful, else False
    """
    common_download_for_higgs_and_flower(collaborators, local_bind_path)


def download_higgs_data(collaborators, local_bind_path):
    """
    Download the data for the model and copy to the respective collaborator workspaces
    Also modify the data.yaml file for all the collaborators
    Args:
        collaborators (list): List of collaborator objects
        local_bind_path (str): Local bind path
    Returns:
        bool: True if successful, else False
    """
    common_download_for_higgs_and_flower(collaborators, local_bind_path)


def common_download_for_higgs_and_flower(collaborators, local_bind_path):
    """
    Common function to download the data for both Higgs and Flower models.
    In future, if the data setup for other models is similar, we can use this function.
    Also, if the setup changes for any of the models, we can modify this function to accommodate the changes.
    """
    log.info(f"Copying {defaults.DATA_SETUP_FILE} from one of the collaborator workspaces to the local bind path..")
    try:
        shutil.copyfile(
            src=os.path.join(collaborators[0].workspace_path, "src", defaults.DATA_SETUP_FILE),
            dst=os.path.join(local_bind_path, defaults.DATA_SETUP_FILE)
        )
    except Exception as e:
        raise ex.DataSetupException(f"Failed to copy data setup file: {e}")

    log.info("Downloading the data for the model. This will take some time to complete based on the data size ..")
    try:
        command = ["python", defaults.DATA_SETUP_FILE, str(len(collaborators))]
        subprocess.run(command, cwd=local_bind_path, check=True)  # nosec B603
    except Exception:
        raise ex.DataSetupException(f"Failed to download data for given model")

    try:
        # Copy the data to the respective workspaces based on the index
        for index, collaborator in enumerate(collaborators, start=1):
            src_folder = os.path.join(local_bind_path, "data", str(index))
            dst_folder = os.path.join(collaborator.workspace_path, "data", str(index))
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                log.info(f"Copied data from {src_folder} to {dst_folder}")
            else:
                raise ex.DataSetupException(f"Source folder {src_folder} does not exist for {collaborator.name}")

            # Modify the data.yaml file for all the collaborators
            collaborator.modify_data_file(
                defaults.COL_DATA_FILE.format(local_bind_path, collaborator.name),
                index,
            )
    except Exception as e:
        raise ex.DataSetupException(f"Failed to modify the data file: {e}")

    # XGBoost model uses folder name higgs_data and Flower model uses data to create data folders.
    shutil.rmtree(os.path.join(local_bind_path, "higgs_data"), ignore_errors=True)
    shutil.rmtree(os.path.join(local_bind_path, "data"), ignore_errors=True)
    return True


def prepare_verifiable_dataset(request, dataset_type):
    """
    Prepare data for S3, Azurite and/or local datasource based on <dataset_type>.
    Args:
        request (object): Pytest request object.
        dataset_type (str): Type of dataset to prepare. Valid values - s3, azure_blob, all.
    """
    if dataset_type not in ["s3", "azure_blob", "all"]:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Valid values are 's3', 'azure_blob', 'all'.")

    num_collaborators = request.config.num_collaborators
    data_path = Path.cwd().absolute() / 'data'
    home_dir = Path().home()
    results_path = os.path.join(home_dir, request.config.results_dir)
    colab_data_mapping = {}

    # Download the histology data and distribute it among collaborators
    # The data is downloaded in the current working directory under 'data' subfolder
    download_histology_data(data_path)
    distribute_data_to_collaborators(num_collaborators, data_path)

    if dataset_type == "all":
        colab_data_mapping = handle_all_dataset_type(num_collaborators, data_path, request)
    else:
        if dataset_type == "s3":
            colab_data_mapping = upload_all_to_s3(num_collaborators, data_path, request)
        elif dataset_type == "azure_blob":
            colab_data_mapping = upload_all_to_azure_blob(num_collaborators, data_path)

    # Create a datasources.json file for each collaborator
    write_datasources_json(num_collaborators, colab_data_mapping, results_path)


def upload_all_to_s3(num_collaborators, data_path, request):
    """Upload all data for each collaborator to S3."""
    colab_data_mapping = {}
    minio_obj = s3_model.MinioServer()

    # Start minio server, create S3 buckets and upload the data to S3
    try:
        if not minio_obj.start_minio_server(
            data_dir=os.path.join(Path().home(), request.config.results_dir, defaults.MINIO_DATA_FOLDER)
        ):
            raise ex.MinioServerStartException(
                "Failed to start minio server. Please check the logs for more details."
            )
    except Exception as e:
        raise ex.MinioServerStartException(
            f"Failed to start minio server. Error: {e}"
        )

    s3_obj = s3_model.S3Bucket()
    for index in range(1, num_collaborators + 1):
        bucket_name = f"col{index}-bucket{index}"
        try:
            s3_obj.create_bucket(bucket_name=bucket_name)
        except Exception as e:
            raise ex.S3BucketCreationException(
                f"Failed to create bucket {bucket_name} for collaborator{index}. Error: {e}"
            )

        collaborator_name = f"collaborator{index}"
        local_dir = data_path / str(index)
        s3_obj.upload_directory(dir_path=local_dir, bucket_name=bucket_name)

        s3_data = {
            "type": "s3",
            "params": {
                "access_key_env_name": "MINIO_ROOT_USER",
                "endpoint": defaults.MINIO_URL,
                "secret_key_env_name": "MINIO_ROOT_PASSWORD",
                "secret_name": "vault_secret_name1",
                "uri": f"s3://{bucket_name}/"
            }
        }
        if collaborator_name not in colab_data_mapping:
            colab_data_mapping[collaborator_name] = {}
        colab_data_mapping[collaborator_name]["s3_data"] = s3_data
        shutil.rmtree(local_dir) # Remove local data after successful upload
    return colab_data_mapping


def upload_all_to_azure_blob(num_collaborators, data_path):
    """Upload all data for each collaborator to Azure Blob (Azurite)."""
    azurite_obj = az_storage_model.AzuriteStorage()
    colab_data_mapping = {}
    try:
        azurite_obj.start_azurite_container()
    except Exception as e:
        raise ex.AzureBlobContainerCreationException(
            f"Failed to start azurite container. Error: {e}"
        )

    # Create container
    for index in range(1, num_collaborators + 1):
        container_name = f"col{index}-container{index}"
        try:
            azurite_obj.create_container(container_name)
            log.info(f"Created container {container_name}")
        except Exception as e:
            if "specified container already exists" in str(e):
                azurite_obj.delete_container(container_name)
                azurite_obj.create_container(container_name)
            else:
                raise ex.AzureBlobContainerCreationException(
                    f"Failed to create container {container_name} for collaborator{index}. Error: {e}"
                )
        collaborator_name = f"collaborator{index}"
        local_dir = data_path / str(index)
        # Upload data to the container
        azurite_obj.upload_data_to_container(
            container_name=container_name,
            data_path=local_dir
        )
        azure_blob_data = {
            "type": "azure_blob",
            "params": {
                "connection_string": azurite_obj.connection_string,
                "container_name": container_name
            }
        }
        if collaborator_name not in colab_data_mapping:
            colab_data_mapping[collaborator_name] = {}
        colab_data_mapping[collaborator_name]["azure_blob_data"] = azure_blob_data
        shutil.rmtree(local_dir)  # Remove local data after successful upload
    return colab_data_mapping


def handle_all_dataset_type(num_collaborators, data_path, request):
    """
    For 'all' dataset_type, split the data into 3 non-overlapping parts and assign to S3, Azure Blob, and local.
    """
    colab_data_mapping = {}

    # Create objects for minio and azurite
    minio_obj = s3_model.MinioServer()
    try:
        if not minio_obj.start_minio_server(
            data_dir=os.path.join(Path().home(), request.config.results_dir, defaults.MINIO_DATA_FOLDER)
        ):
            raise ex.MinioServerStartException(
                "Failed to start minio server. Please check the logs for more details."
            )
    except Exception as e:
        raise ex.MinioServerStartException(
            f"Failed to start minio server. Error: {e}"
        )

    s3_obj = s3_model.S3Bucket()

    azurite_obj = az_storage_model.AzuriteStorage()
    try:
        azurite_obj.start_azurite_container()
    except Exception as e:
        raise ex.AzureBlobContainerCreationException(
            f"Failed to start azurite container. Error: {e}"
        )

    # Upload data to S3, Azure Blob and local for each collaborator
    for index in range(1, num_collaborators + 1):
        collaborator_name = f"collaborator{index}"
        local_dir = data_path / str(index)
        all_files = sorted([f for f in local_dir.iterdir() if f.is_dir() or f.is_file()])

        total = len(all_files)
        split_size = total // 3
        splits = [
            all_files[:split_size],
            all_files[split_size:2*split_size],
            all_files[2*split_size:]
        ]

        # Prepare temp dirs for each split
        s3_dir = local_dir / "s3_part"
        azure_dir = local_dir / "azure_part"
        local_part_dir = local_dir / "local_part"
        for d in [s3_dir, azure_dir, local_part_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Move files to their respective dirs
        for f in splits[0]:
            shutil.move(str(f), s3_dir / f.name)
        for f in splits[1]:
            shutil.move(str(f), azure_dir / f.name)
        for f in splits[2]:
            shutil.move(str(f), local_part_dir / f.name)

        # Ensure each part has at least one folder (copy from the largest part if needed)
        part_dirs = [s3_dir, azure_dir, local_part_dir]
        part_counts = [len(list(d.iterdir())) for d in part_dirs]
        if any(count == 0 for count in part_counts):
            # Find the largest part
            largest_idx = part_counts.index(max(part_counts))
            largest_dir = part_dirs[largest_idx]
            largest_files = list(largest_dir.iterdir())
            for idx, count in enumerate(part_counts):
                if count == 0 and largest_files:
                    # Copy (not move) the first folder/file from the largest part
                    src = largest_files[0]
                    dst = part_dirs[idx] / src.name
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

        # S3 data
        bucket_name = f"col{index}-bucket{index}"
        s3_obj.create_bucket(bucket_name=bucket_name)
        s3_obj.upload_directory(dir_path=s3_dir, bucket_name=bucket_name)
        s3_data = {
            "type": "s3",
            "params": {
                "access_key_env_name": "MINIO_ROOT_USER",
                "endpoint": defaults.MINIO_URL,
                "secret_key_env_name": "MINIO_ROOT_PASSWORD",
                "secret_name": "vault_secret_name1",
                "uri": f"s3://{bucket_name}/"
            }
        }

        # Azure Blob data
        container_name = f"col{index}-container{index}"
        azurite_obj.create_container(container_name)
        azurite_obj.upload_data_to_container(container_name=container_name, data_path=azure_dir)
        azure_blob_data = {
            "type": "azure_blob",
            "params": {
                "connection_string": azurite_obj.connection_string,
                "container_name": container_name
            }
        }

        # Local data
        local_data = {
            "type": "local",
            "params": {
                "path": str(local_part_dir.relative_to(Path.cwd()))
            }
        }
        # Print local data objects count
        log.info(f"Retained {len(list(local_part_dir.rglob('*')))} files in local data for {collaborator_name}")
        colab_data_mapping[collaborator_name] = {
            "s3_data": s3_data,
            "azure_blob_data": azure_blob_data,
            "local_data": local_data
        }
        # Clean up temp dirs after upload if needed
        # shutil.rmtree(s3_dir)
        # shutil.rmtree(azure_dir)
        # local_part_dir is kept for local access

    return colab_data_mapping


def write_datasources_json(num_collaborators, colab_data_mapping, results_path):
    """
    Create a datasources.json file for each collaborator.
    Args:
        num_collaborators (int): Number of collaborators.
        colab_data_mapping (dict): Mapping of collaborator names to their data sources.
        results_path (str): Path to the results directory.
    """
    for index in range(1, num_collaborators + 1):
        collaborator_name = f"collaborator{index}"
        col_mapping = colab_data_mapping[collaborator_name]
        combined_data = {}

        # Add s3_data as s3_ds1
        if "s3_data" in col_mapping:
            combined_data["s3_ds1"] = col_mapping["s3_data"]

        # Add azure_blob_data as azure_ds0
        if "azure_blob_data" in col_mapping:
            combined_data["azure_ds1"] = col_mapping["azure_blob_data"]

        # Add local_data as local_ds1
        if "local_data" in col_mapping:
            combined_data["local_ds1"] = col_mapping["local_data"]

        ds_file = os.path.join(results_path, "datasources", collaborator_name, "datasources.json")
        os.makedirs(os.path.dirname(ds_file), exist_ok=True)

        with open(ds_file, "w") as f:
            json.dump(combined_data, f, indent=2)


def distribute_data_to_collaborators(num_collaborators, data_path):
    """
    Distribute the data among the collaborators uniformly.
    Example: Assuming num_collaborators is 3
        If data_path has folder Kather_texture_2016_image_tiles_5000 (torch/histology) which further has 8 subfolders,
        then the data will be distributed as:
            collaborator1: 1 / first 3 subfolders
            collaborator2: 2 / next 3 subfolders
            collaborator3: 3 / last 2 subfolders
        If data_path itself has multiple folders say 8, then the data will be distributed as:
            collaborator1: 1 / first 3 folders
            collaborator2: 2 / next 3 folders
            collaborator3: 3 / last 2 folders
    Args:
        num_collaborators (int): Number of collaborators.
        data_path (str): Path to the data directory.
    Raises:
        Exception: If the data distribution fails.
    """
    # Pre-check: skip if all collaborator folders exist and are non-empty
    already_distributed = True
    for index in range(1, num_collaborators + 1):
        collaborator_data_path = data_path / str(index)
        if not (collaborator_data_path.exists() and any(collaborator_data_path.iterdir())):
            already_distributed = False
            break

    # If already distributed, just collect the mapping and return
    if already_distributed:
        log.info("Data already distributed among collaborators. Skipping distribution.")
        return

    # If data_path has only one folder, go inside it and use its subfolders
    all_entries = [f for f in data_path.iterdir() if f.is_dir()]
    if len(all_entries) == 1:
        # Use subfolders inside the single folder
        all_folders = [f for f in all_entries[0].iterdir() if f.is_dir()]
    else:
        all_folders = all_entries
    all_folders.sort()  # For deterministic split

    num_folders = len(all_folders)
    folders_per_collab = [num_folders // num_collaborators] * num_collaborators
    for i in range(num_folders % num_collaborators):
        folders_per_collab[i] += 1

    start = 0
    for index in range(1, num_collaborators + 1):
        collaborator_data_path = data_path / str(index)
        collaborator_data_path.mkdir(parents=True, exist_ok=True)
        end = start + folders_per_collab[index - 1]
        for folder in all_folders[start:end]:
            dest = collaborator_data_path / folder.name
            if folder.parent != collaborator_data_path:
                folder.rename(dest)
        start = end

    # Remove all files/folders from 'data' except collaborator folders (1, 2, 3, ...)
    for entry in data_path.iterdir():
        if entry.is_dir() and entry.name not in [str(i) for i in range(1, num_collaborators + 1)]:
            shutil.rmtree(entry)
            log.info(f"Removed folder {entry} from data path")
        elif entry.is_file() and entry.name.endswith(".zip"):
            os.remove(entry)
            log.info(f"Removed zip file {entry} from data path")


def download_histology_data(data_path):
    """
    Download the histology data using its dataloader module.
    The data is downloaded in the current working directory under 'data' subfolder.
    """
    # Check if data already exists, if yes delete the folder and download again
    if data_path.exists() and any(data_path.iterdir()):
        log.info("Data already exists. Deleting the folder and downloading again..")
        shutil.rmtree(data_path)

    # Import the dataloader module for torch/histology to download the data
    # As the folder name contains hyphen, we need to use importlib to import the module
    dataloader_module = importlib.import_module("openfl-workspace.torch.histology.src.dataloader")

    # Download the data for torch/histology in current folder as internally it uses the current folder as data path
    try:
        log.info(f"Downloading data for {defaults.ModelName.TORCH_HISTOLOGY_S3.value}")
        dataloader_module.HistologyDataset()
        log.info("Download completed")
    except Exception as e:
        raise ex.DataDownloadException(
            f"Failed to download data for {defaults.ModelName.TORCH_HISTOLOGY_S3.value}. Error: {e}"
        )
