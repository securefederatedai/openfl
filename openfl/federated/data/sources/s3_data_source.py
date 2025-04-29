# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from typing import Generator
from urllib.parse import urlparse

import boto3

from openfl.federated.data.sources.data_source import DataSource, DataSourceType


class S3DataSource(DataSource):
    """Class for S3 data"""

    def __init__(
        self,
        uri: str,
        endpoint=None,
        access_key_env_name=None,
        secret_key_env_name=None,
        secret_name=None,
        hash_func=None,
    ):
        super().__init__(DataSourceType.S3)
        self.uri = uri
        self.endpoint = endpoint
        self.access_key_env_name = access_key_env_name
        self.secret_key_env_name = secret_key_env_name
        self.secret_name = secret_name
        self.hash_func = hash_func

        access_key_id = os.environ.get(access_key_env_name) if access_key_env_name else None
        secret_access_key = os.environ.get(secret_key_env_name) if secret_key_env_name else None

        self._s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def enumerate_files(self) -> Generator[str, None, None]:
        """Enumerate all files in the data source"""
        parsed = urlparse(self.uri)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")  # Remove leading slash

        paginator = self._s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    obj_key = obj["Key"]
                    if obj_key.endswith("/"):  # Ignore directories
                        continue
                    yield f"s3://{bucket_name}/{obj_key}"

    def _get_s3_etag(self, obj_path: str):
        parsed = urlparse(obj_path)
        bucket_name = parsed.netloc
        object_key = parsed.path.lstrip("/")
        response = self._s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return response["ETag"]

    def _read_s3_object(self, obj_path: str):
        parsed = urlparse(obj_path)
        bucket_name = parsed.netloc
        object_key = parsed.path.lstrip("/")
        response = self._s3_client.get_object(Bucket=bucket_name, Key=object_key)
        return response["Body"].read()

    def compute_file_hash(self, path: str) -> str:
        if not self.hash_func:
            return self._get_s3_etag(path)
        else:
            data = self._read_s3_object(path)
            return self.hash_func(data).hexdigest()

    def read_blob(self, path):
        return self._read_s3_object(path)

    @classmethod
    def from_dict(cls, ds_dict: dict):
        hash_func_name = ds_dict.get("hash_func", None)
        if hash_func_name:
            hash_func = getattr(hashlib, hash_func_name, None)
        else:
            hash_func = None
        return cls(
            uri=ds_dict["uri"],
            endpoint=ds_dict.get("endpoint", None),
            access_key_env_name=ds_dict.get("access_key_env_name", None),
            secret_key_env_name=ds_dict.get("secret_key_env_name", None),
            hash_func=hash_func,
        )
