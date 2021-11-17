# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Histology Shard Descriptor."""

import logging
import os
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.utilities import tqdm_report_hook
from openfl.utilities import validate_file_hash


logger = logging.getLogger(__name__)


class HistologyShardDataset(ShardDataset):
    """Histology shard dataset class."""

    TRAIN_SPLIT_RATIO = 0.8

    def __init__(self, data_folder: Path, data_type='train', rank=1, worldsize=1):
        """Histology shard dataset class."""
        self.data_type = data_type
        root = Path(data_folder) / 'Kather_texture_2016_image_tiles_5000'
        classes = [d.name for d in root.iterdir() if d.is_dir()]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.samples = []
        root = root.expanduser()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            for class_root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(class_root, fname)
                    item = path, class_index
                    self.samples.append(item)

        idx_range = list(range(len(self.samples)))
        idx_sep = int(len(idx_range) * HistologyShardDataset.TRAIN_SPLIT_RATIO)
        train_idx, test_idx = np.split(idx_range, [idx_sep])
        if data_type == 'train':
            self.idx = train_idx[rank - 1::worldsize]
        else:
            self.idx = test_idx[rank - 1::worldsize]

    def __len__(self) -> int:
        """Return the len of the shard dataset."""
        return len(self.idx)

    def load_pil(self, path):
        """Load image."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index: int) -> Tuple['Image', int]:
        """Return an item by the index."""
        path, target = self.samples[self.idx[index]]
        sample = self.load_pil(path)
        return sample, target


class HistologyShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    URL = ('https://zenodo.org/record/53169/files/Kather_'
           'texture_2016_image_tiles_5000.zip?download=1')
    FILENAME = 'Kather_texture_2016_image_tiles_5000.zip'
    ZIP_SHA384 = ('7d86abe1d04e68b77c055820c2a4c582a1d25d2983e38ab724e'
                  'ac75affce8b7cb2cbf5ba68848dcfd9d84005d87d6790')
    DEFAULT_PATH = Path.home() / '.openfl' / 'data'

    def __init__(
            self,
            data_folder: Path = DEFAULT_PATH,
            rank_worldsize: str = '1,1',
            **kwargs
    ):
        """Initialize HistologyShardDescriptor."""
        self.data_folder = Path.cwd() / data_folder
        self.download_data()
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

    def download_data(self):
        """Download prepared shard dataset."""
        os.makedirs(self.data_folder, exist_ok=True)
        filepath = self.data_folder / HistologyShardDescriptor.FILENAME
        if not filepath.exists():
            reporthook = tqdm_report_hook()
            urlretrieve(HistologyShardDescriptor.URL, filepath, reporthook)  # nosec
            validate_file_hash(filepath, HistologyShardDescriptor.ZIP_SHA384)
            with ZipFile(filepath, 'r') as f:
                f.extractall(self.data_folder)

    def get_dataset(self, dataset_type):
        """Return a shard dataset by type."""
        return HistologyShardDataset(
            data_folder=self.data_folder,
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        shape = self.get_dataset('train')[0][0].size
        return [str(dim) for dim in shape]

    @property
    def target_shape(self):
        """Return the target shape info."""
        target = self.get_dataset('train')[0][1]
        shape = np.array([target]).shape
        return [str(dim) for dim in shape]

    @property
    def dataset_description(self) -> str:
        """Return the shard dataset description."""
        return (f'Histology dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
