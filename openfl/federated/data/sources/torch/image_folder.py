# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io

from PIL import Image

from openfl.federated.data.sources.data_source import DataSource
from openfl.federated.data.sources.torch.folder_dataset import FolderDataset, LabelMapper


class ImageFolder(FolderDataset):
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
        raw_data = self.datasource.read_blob(file_path)
        return Image.open(io.BytesIO(raw_data))
