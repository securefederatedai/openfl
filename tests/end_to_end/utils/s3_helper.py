# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
import logging
from pathlib import Path

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.exceptions as ex

log = logging.getLogger(__name__)


def create_collaborator_datasource_json(colab_bucket_mapping, endpoint=constants.MINIO_URL):
    """
    Create a datasources.json file for a collaborator.

    Args:
        colab_bucket_mapping (dict): Mapping of given collaborator with its datasources
        endpoint (str): S3 endpoint URL

    Returns:
        JSON object: JSON object representing the datasource configuration
    """
    collaborator_name = colab_bucket_mapping["collaborator"]
    buckets = colab_bucket_mapping["buckets"]
    local_data_path = colab_bucket_mapping["local_data_path"]
    index = int(''.join(filter(str.isdigit, collaborator_name)))
    data = {}

    for i, bucket in enumerate(buckets, 1):
        ds_key = f"s3_ds{i}"
        data[ds_key] = {
            "type": "s3",
            "params": {
                "access_key_env_name": "MINIO_ROOT_USER",
                "endpoint": endpoint,
                "secret_key_env_name": "MINIO_ROOT_PASSWORD",
                "secret_name": f"vault_secret_name{i}",
                "uri": f"s3://{bucket}/"
            }
        }
    # Add local datasource for odd collaborators (collaborator index is odd)
    if index is not None and index % 2 == 1:
        data[f"local_ds{index}"] = {
            "type": "local",
            "params": {
                "path": str(Path(local_data_path).relative_to(Path.cwd()))
            }
        }

    return data


def upload_data_to_s3(s3_obj, colab_bucket_mapping_list):
    """
    Upload data to S3 buckets based on the provided mapping.
    Args:
        s3_obj (S3Helper): S3Helper object for S3 operations
        colab_bucket_mapping_list (list): List of dictionaries containing collaborator and bucket mapping
    Returns:
        bool: True if upload was successful, raises DataUploadToS3Exception exception otherwise
    """
    for colab in colab_bucket_mapping_list:
        folder_path = Path(colab["local_data_path"])
        buckets = colab["buckets"]
        if len(buckets) == 2:
            # Split the folder contents equally for two buckets
            all_items = sorted([item for item in folder_path.iterdir() if item.is_dir() or item.is_file()])
            mid = len(all_items) // 2
            split_items = [all_items[:mid], all_items[mid:]]
            for i, bucket_name in enumerate(buckets):
                temp_dir = folder_path / f"tmp_upload_{i+1}"
                temp_dir.mkdir(exist_ok=True)
                for item in split_items[i]:
                    dest = temp_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                try:
                    s3_obj.upload_directory(dir_path=temp_dir, bucket_name=bucket_name)
                    log.info(f"Uploaded data to bucket {bucket_name} from {temp_dir}")
                except Exception as e:
                    raise ex.DataUploadToS3Exception(
                        f"Failed to upload data to bucket {bucket_name}. Error: {e}"
                    )
                shutil.rmtree(temp_dir)
        else:
            # Only one bucket, upload the whole folder
            bucket_name = buckets[0]
            try:
                s3_obj.upload_directory(dir_path=folder_path, bucket_name=bucket_name)
                log.info(f"Uploaded data to bucket {bucket_name} from {folder_path}")
            except Exception as e:
                raise ex.DataUploadToS3Exception(
                    f"Failed to upload data to bucket {bucket_name}. Error: {e}"
                )
    return True
