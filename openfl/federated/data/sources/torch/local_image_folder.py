# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Union

from PIL import Image

from openfl.federated.data.sources.torch.local_folder import LocalFolder


class LocalImageFolder(LocalFolder):
    """Custom dataset that loads all images from a local directory tree and assigns labels
    based on folder names."""

    def pil_loader(self, path: Union[str, Path]) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def load_file(self, file_path):
        """Load an image file from the dataset."""
        return self.pil_loader(file_path)
