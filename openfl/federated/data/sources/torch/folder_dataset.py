# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any, Dict

import torch


class LabelMapper:
    """
    A utility class for mapping class names (labels) to unique integer indices and vice versa.
    This ensures consistent labeling across data sources and processes by maintaining a
    bidirectional mapping between labels and their corresponding indices.
    """

    def __init__(self):
        self.label_to_idx = {}
        self.idx_to_label = {}

    def get_label_index(self, label: str) -> int:
        """Assigns or retrieves the index of a label."""
        if label not in self.label_to_idx:
            new_index = len(self.label_to_idx)
            self.label_to_idx[label] = new_index
            self.idx_to_label[new_index] = label
        return self.label_to_idx[label]

    def get_label_name(self, index: int) -> str:
        """Retrieves the original label name from an index."""
        return self.idx_to_label.get(index, None)


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, label_mapper: LabelMapper, transform=None):
        """
        Args:
            label_mapper (LabelMapper): LabelMapper object to map class names to indices.
            transform (callable, optional): Transformations to apply to images.
        """
        self.transform = transform
        self.label_mapper = label_mapper

        # Build the dataset
        self.samples = self._load_samples()

    def _get_label(self, file_path):
        """Get the label for a given file path."""
        label_name = file_path.split("/")[-2] if len(file_path.split("/")) > 1 else None
        return self.label_mapper.get_label_index(label_name)

    def _load_samples(self):
        """Loads all file paths and their inferred labels"""
        return [
            (file_path, self._get_label(file_path))
            for file_path in self.datasource.enumerate_files()
        ]

    @abstractmethod
    def load_file(self, file_path):
        """Load a file from the dataset."""
        pass

    def __getitem__(self, index) -> Dict[str, Any]:
        file_path, label = self.samples[index]
        file_data = self.load_file(str(file_path))

        if self.transform:
            file_data = self.transform(file_data)

        return {"data": file_data, "label": label, "path": str(file_path)}

    def __len__(self):
        return len(self.samples)
