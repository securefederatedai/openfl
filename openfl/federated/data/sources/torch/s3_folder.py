# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from urllib.parse import urlparse

import boto3

from openfl.federated.data.sources.torch.folder_dataset import FolderDataset, LabelMapper


class S3Folder(FolderDataset):
    def __init__(
        self,
        uri,
        label_mapper: LabelMapper,
        endpoint=None,
        access_key_env_name=None,
        secret_key_env_name=None,
        transform=None,
    ):
        """
        Args:
            uri (str): URI to the S3 object.
            label_mapper (LabelMapper): LabelMapper object to map class names to indices.
            endpoint (str, optional): S3 endpoint URL.
            access_key_env_name (str, optional): Environment variable name for S3 access key.
            secret_key_env_name (str, optional): Environment variable name for S3 secret key.
            transform (callable, optional): Transformations to apply to images.
        """
        self.uri = uri
        self.endpoint = endpoint
        self.access_key_env_name = access_key_env_name
        self.secret_key_env_name = secret_key_env_name
        super().__init__(label_mapper, transform=transform)

    def _load_samples(self):
        """Loads all file paths and their inferred labels into self.samples."""
        parsed = urlparse(self.uri)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")
        access_key = os.environ.get(self.access_key_env_name) if self.access_key_env_name else None
        secret_key = os.environ.get(self.secret_key_env_name) if self.secret_key_env_name else None

        s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        paginator = s3_client.get_paginator("list_objects_v2")
        samples = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    obj_key = obj["Key"]
                    if obj_key.endswith("/"):  # Ignore directories
                        continue

                    # Extract label from parent directory
                    parts = obj_key.split("/")
                    if len(parts) > 1:
                        label = parts[-2]
                    else:
                        label = None

                    obj_path = f"s3://{bucket_name}/{obj_key}"
                    label_idx = self.label_mapper.get_label_index(label)  # Use common mapping
                    samples.append((obj_path, label_idx))
        return samples
