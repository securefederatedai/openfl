# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
from urllib.parse import urlparse

import boto3
from PIL import Image

from openfl.federated.data.sources.torch.s3_folder import S3Folder


class S3ImageFolder(S3Folder):
    """A subclass of S3Folder specifically for loading images."""

    def load_file(self, obj_path: str) -> Image.Image:
        """Loads an image from the given S3 path and returns a PIL image."""
        parsed = urlparse(obj_path)
        bucket_name = parsed.netloc
        object_key = parsed.path.lstrip("/")

        access_key = os.environ.get(self.access_key_env_name) if self.access_key_env_name else None
        secret_key = os.environ.get(self.secret_key_env_name) if self.secret_key_env_name else None

        s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)

        img_data = response["Body"].read()
        return Image.open(io.BytesIO(img_data))
