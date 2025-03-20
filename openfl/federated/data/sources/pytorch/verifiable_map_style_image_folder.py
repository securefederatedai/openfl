# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.federated.data.sources.data_source import DataSourceType
from openfl.federated.data.sources.pytorch.local_folder import LabelMapper
from openfl.federated.data.sources.pytorch.local_image_folder import LocalImageFolder
from openfl.federated.data.sources.pytorch.verifiable_map_style_dataset import (
    VerifiableMapStyleDataset,
)


class VerifiableImageFolder(VerifiableMapStyleDataset):
    """VerifiableImageFolder class for image folder datasets."""

    def __init__(self, vds, transform=None, verify_dataset=False):
        self.label_mapper = LabelMapper()
        super().__init__(vds, transform=transform, verify_dataset=verify_dataset)

    def create_datasets(self):
        datasources = []
        for data_source in self.verifiable_dataset_info.data_sources:
            if data_source.type == DataSourceType.LOCAL:
                datasource_full_path = (
                    self.verifiable_dataset_info.base_path / data_source.source_path
                )
                datasources.append(
                    LocalImageFolder(
                        datasource_full_path, self.label_mapper, transform=self.transform
                    )
                )
            else:
                raise ValueError(f"Unknown or unsupported storage type: {data_source.type}")
        return datasources
