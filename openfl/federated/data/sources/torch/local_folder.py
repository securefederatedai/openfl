# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset


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


class LocalFolder(Dataset):
    def __init__(self, base_path, label_mapper: LabelMapper, transform=None):
        """
        Args:
            base_path (str or Path): Root directory containing labeled subdirectories.
            label_mapper (LabelMapper): LabelMapper object to map class names to indices.
            transform (callable, optional): Transformations to apply to loaded data.
        """
        self.base_path = Path(base_path).resolve()
        self.transform = transform
        self.samples = []
        self.label_mapper = label_mapper

        # Build the dataset
        self._load_samples()

    def _load_samples(self):
        """Recursively find all files and assign labels based on the directory name."""
        for file_path in self.base_path.rglob("*"):  # Search for all files in subdirectories
            if file_path.is_file():
                # Get parent directory as label
                label_name = file_path.parent.name
                label_idx = self.label_mapper.get_label_index(label_name)  # Use common mapping
                self.samples.append((file_path, label_idx))

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
