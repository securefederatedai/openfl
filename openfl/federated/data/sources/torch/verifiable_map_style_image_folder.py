# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.federated.data.sources.torch.folder_dataset import LabelMapper
from openfl.federated.data.sources.torch.image_folder import ImageFolder
from openfl.federated.data.sources.torch.verifiable_map_style_dataset import (
    VerifiableMapStyleDataset,
)


class VerifiableImageFolder(VerifiableMapStyleDataset):
    """VerifiableImageFolder class for image folder datasets."""

    def __init__(self, vds, transform=None, verify_dataset_items=False):
        self.label_mapper = LabelMapper()
        super().__init__(vds, transform=transform, verify_dataset_items=verify_dataset_items)

    def create_datasets(self):
        datasources = []
        for data_source in self.verifiable_dataset_info.data_sources:
            datasources.append(
                ImageFolder(data_source, self.label_mapper, transform=self.transform)
            )
        return datasources
