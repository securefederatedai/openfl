# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TinyImageNet Shard Descriptor."""

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class TinyImageNetDataset(ShardDataset):
    """TinyImageNet shard dataset class."""

    NUM_IMAGES_PER_CLASS = 500

    def __init__(self, data_folder: Path, data_type='train', rank=1, worldsize=1):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self._common_data_folder = data_folder
        self._data_folder = os.path.join(data_folder, data_type)
        self.labels = {}  # fname - label number mapping
        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self._data_folder, '**', '*.JPEG'),
                recursive=True
            )
        )[rank - 1::worldsize]
        wnids_path = os.path.join(self._common_data_folder, 'wnids.txt')
        with open(wnids_path, 'r', encoding='utf-8') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}
        self.fill_labels()

    def __len__(self) -> int:
        """Return the len of the shard dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple['Image', int]:
        """Return an item by the index."""
        file_path = self.image_paths[index]
        label = self.labels[os.path.basename(file_path)]
        return self.read_image(file_path), label

    def read_image(self, path: Path) -> Image:
        """Read the image."""
        img = Image.open(path)
        return img

    def fill_labels(self) -> None:
        """Fill labels."""
        if self.data_type == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels[f'{label_text}_{cnt}.JPEG'] = i
        elif self.data_type == 'val':
            val_annotations_path = os.path.join(self._data_folder, 'val_annotations.txt')
            with open(val_annotations_path, 'r', encoding='utf-8') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]


class TinyImageNetShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(
            self,
            data_folder: str = 'data',
            rank_worldsize: str = '1,1',
            **kwargs
    ):
        """Initialize TinyImageNetShardDescriptor."""
        self.common_data_folder = Path.cwd() / data_folder
        self.data_folder = Path.cwd() / data_folder / 'tiny-imagenet-200'
        self.download_data()
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

    def download_data(self):
        """Download prepared shard dataset."""
        zip_file_path = self.common_data_folder / 'tiny-imagenet-200.zip'
        os.makedirs(self.common_data_folder, exist_ok=True)
        os.system(f'wget --no-clobber http://cs231n.stanford.edu/tiny-imagenet-200.zip'
                  f' -O {zip_file_path}')
        shutil.unpack_archive(str(zip_file_path), str(self.common_data_folder))

    def get_dataset(self, dataset_type):
        """Return a shard dataset by type."""
        return TinyImageNetDataset(
            data_folder=self.data_folder,
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['64', '64', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['64', '64']

    @property
    def dataset_description(self) -> str:
        """Return the shard dataset description."""
        return (f'TinyImageNetDataset dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
