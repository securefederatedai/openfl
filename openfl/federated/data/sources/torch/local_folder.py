# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from openfl.federated.data.sources.torch.folder_dataset import FolderDataset, LabelMapper


class LocalFolder(FolderDataset):
    def __init__(self, base_path, label_mapper: LabelMapper, transform=None):
        """
        Args:
            base_path (str or Path): Root directory containing labeled subdirectories.
            label_mapper (LabelMapper): LabelMapper object to map class names to indices.
            transform (callable, optional): Transformations to apply to loaded data.
        """
        self.base_path = Path(base_path).resolve()
        super().__init__(label_mapper, transform=transform)

    def _load_samples(self):
        """Recursively find all files and assign labels based on the directory name."""
        samples = []
        for file_path in self.base_path.rglob("*"):  # Search for all files in subdirectories
            if file_path.is_file():
                # Get parent directory as label
                label_name = file_path.parent.name
                label_idx = self.label_mapper.get_label_index(label_name)  # Use common mapping
                samples.append((file_path, label_idx))
        return samples
