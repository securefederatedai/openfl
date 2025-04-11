# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import numpy as np
import torch

from openfl.federated.data.sources.verifiable_dataset_info import (
    VerifiableDatasetInfo,
)


class VerifiableMapStyleDataset(torch.utils.data.Dataset):
    """Base class for different types of map style datasets."""

    def __init__(self, vds: VerifiableDatasetInfo, transform=None, verify_dataset_items=False):
        self.transform = transform
        self.verifiable_dataset_info = vds
        self.verify_dataset_items = verify_dataset_items
        self.datasets = self.create_datasets()

        # create indices for fast lookup
        lengths = list(map(len, self.datasets))
        self.cumulative_sizes = np.cumsum(lengths)

    def __getitem__(self, idx):
        # find which sub-dataset this index belongs to
        dataset_idx = self.cumulative_sizes.searchsorted(idx, side="right")
        # find the data in that sub-dataset
        data_idx = idx - self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else idx
        item = self.datasets[dataset_idx][data_idx]
        data_path = item["path"]

        if self.verify_dataset_items:
            item_hash = self.verifiable_dataset_info.data_sources[dataset_idx].compute_file_hash(
                data_path
            )
            if not self.verifiable_dataset_info.verify_single_file(data_path, item_hash):
                raise ValueError(f"Data integrity check failed for {data_path}")

        return item["data"], item["label"]

    def __len__(self):
        return self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0

    @abstractmethod
    def create_datasets(self):
        pass
