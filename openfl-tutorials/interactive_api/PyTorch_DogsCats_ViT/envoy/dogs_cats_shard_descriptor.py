# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cats and dogs shard descriptor."""

import os
from pathlib import Path
from random import shuffle

import numpy as np
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class DogsCatsShardDataset(ShardDataset):
    """Dogs and cats Shard dataset class."""

    def __init__(self, data_type: str, dataset_dir: Path,
                 rank=1, worldsize=1, enforce_image_hw=None):
        """Initialize DogsCatsShardDataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.dataset_dir = dataset_dir
        self.enforce_image_hw = enforce_image_hw
        self.img_path = self.dataset_dir / data_type

        self.img_names = [
            img_name
            for img_name in sorted(os.listdir(self.img_path))
            if img_name[-3:] == 'jpg'
        ]

        # Sharding
        self.img_names = self.img_names[self.rank - 1::self.worldsize]
        # Shuffling the results dataset after choose half pictures of each class
        shuffle(self.img_names)

    def __getitem__(self, index):
        """Return a item by the index."""
        name = self.img_names[index]
        # Reading data
        img = Image.open(self.img_path / name)
        img_class = 1 if name[:3] == 'dog' else 0
        assert name[:3] in {'cat', 'dog'}, 'Wrong object classification'

        if self.enforce_image_hw is not None:
            # If we need to resize data
            # PIL accepts (w,h) tuple, not (h,w)
            img = img.resize(self.enforce_image_hw[::-1])

        img = np.asarray(img)

        assert img.shape[2] == 3

        return img, np.asarray([img_class], dtype=np.uint8)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.img_names)


class DogsCatsShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, data_folder: str = 'data',
                 rank_worldsize: str = '1,3',
                 enforce_image_hw: str = None) -> None:
        """Initialize DogsCatsShardDescriptor."""
        super().__init__()
        # Settings for sharding the dataset
        self.rank, self.worldsize = map(lambda x: int(x), rank_worldsize.split(','))

        self.data_folder = Path.cwd() / data_folder

        # Settings for resizing data
        self.enforce_image_hw = None
        if enforce_image_hw is not None:
            self.enforce_image_hw = tuple(map(lambda x: int(x), enforce_image_hw.split(',')))

        # Calculating data and target shapes
        ds = self.get_dataset()
        sample, target = ds[0]
        self._sample_shape = [str(dim) for dim in sample.shape]
        self._target_shape = [str(*target.shape)]

        assert self._target_shape[0] == '1', 'Target shape Error'

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        return DogsCatsShardDataset(
            data_type=dataset_type,
            dataset_dir=self.data_folder,
            rank=self.rank,
            worldsize=self.worldsize,
            enforce_image_hw=self.enforce_image_hw
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return self._sample_shape

    @property
    def target_shape(self):
        """Return the target shape info."""
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Dogs and Cats dataset, shard number {self.rank} '
                f'out of {self.worldsize}')
