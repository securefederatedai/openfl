# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Landmarks Shard Descriptor."""

import json
import shutil
from hashlib import md5
from logging import getLogger
from pathlib import Path
from random import shuffle
from typing import Dict
from typing import List
from zipfile import ZipFile

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = getLogger(__name__)


class LandmarkShardDataset(ShardDataset):
    """Landmark Shard dataset class."""

    def __init__(self, dataset_dir: Path,
                 rank: int = 1, worldsize: int = 1) -> None:
        """Initialize LandmarkShardDataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.dataset_dir = dataset_dir
        self.img_names = list(self.dataset_dir.glob('img_*.npy'))

        # Sharding
        self.img_names = self.img_names[self.rank - 1::self.worldsize]
        # Shuffling the results dataset after choose half pictures of each class
        shuffle(self.img_names)

    def __getitem__(self, index) -> np.ndarray:
        """Return a item by the index."""
        # Get name key points file
        # f.e. image name:  'img_123.npy, corresponding name of the key points: 'keypoints_123.npy'
        kp_name = str(self.img_names[index]).replace('img', 'keypoints')
        return np.load(self.img_names[index]), np.load(self.dataset_dir / kp_name)

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.img_names)


class LandmarkShardDescriptor(ShardDescriptor):
    """Landmark Shard descriptor class."""

    def __init__(self, data_folder: str = 'data',
                 rank_worldsize: str = '1, 1',
                 **kwargs) -> None:
        """Initialize LandmarkShardDescriptor."""
        super().__init__()
        # Settings for sharding the dataset
        self.rank, self.worldsize = map(int, rank_worldsize.split(','))

        self.data_folder = Path.cwd() / data_folder
        self.download_data()

        # Calculating data and target shapes
        ds = self.get_dataset()
        sample, target = ds[0]
        self._sample_shape = [str(dim) for dim in sample.shape]
        self._target_shape = [str(len(target.shape))]

        if self._target_shape[0] != '1':
            raise ValueError('Target has a wrong shape')

    def process_data(self, name_csv_file) -> None:
        """Process data from csv to numpy format and save it in the same folder."""
        data_df = pd.read_csv(self.data_folder / name_csv_file)
        data_df.fillna(method='ffill', inplace=True)
        keypoints = data_df.drop('Image', axis=1)
        cur_folder = self.data_folder.relative_to(Path.cwd())

        for i in range(data_df.shape[0]):
            img = data_df['Image'][i].split(' ')
            img = np.array(['0' if x == '' else x for x in img], dtype='float32').reshape(96, 96)
            np.save(str(cur_folder / f'img_{i}.npy'), img)
            y = np.array(keypoints.iloc[i, :], dtype='float32')
            np.save(str(cur_folder / f'keypoints_{i}.npy'), y)

    def download_data(self) -> None:
        """Download dataset from Kaggle."""
        if self.is_dataset_complete():
            return

        self.data_folder.mkdir(parents=True, exist_ok=True)

        logger.info('Your dataset is absent or damaged. Downloading ... ')
        api = KaggleApi()
        api.authenticate()

        if Path('data').exists():
            shutil.rmtree('data')

        api.competition_download_file(
            'facial-keypoints-detection',
            'training.zip', path=self.data_folder
        )

        with ZipFile(self.data_folder / 'training.zip', 'r') as zipobj:
            zipobj.extractall(self.data_folder)

        (self.data_folder / 'training.zip').unlink()

        self.process_data('training.csv')
        (self.data_folder / 'training.csv').unlink()
        self.save_all_md5()

    def get_dataset(self, dataset_type='train') -> LandmarkShardDataset:
        """Return a shard dataset by type."""
        return LandmarkShardDataset(
            dataset_dir=self.data_folder,
            rank=self.rank,
            worldsize=self.worldsize
        )

    def calc_all_md5(self) -> Dict[str, str]:
        """Calculate hash of all dataset."""
        md5_dict = {}
        for root in self.data_folder.glob('*.npy'):
            md5_calc = md5(usedforsecurity=False)
            rel_file = root.relative_to(self.data_folder)

            with open(self.data_folder / rel_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_calc.update(chunk)
                md5_dict[str(rel_file)] = md5_calc.hexdigest()
        return md5_dict

    def save_all_md5(self) -> None:
        """Save dataset hash."""
        all_md5 = self.calc_all_md5()
        with open(self.data_folder / 'dataset.json', 'w', encoding='utf-8') as f:
            json.dump(all_md5, f)

    def is_dataset_complete(self) -> bool:
        """Check dataset integrity."""
        dataset_md5_path = self.data_folder / 'dataset.json'
        if dataset_md5_path.exists():
            with open(dataset_md5_path, 'r', encoding='utf-8') as f:
                old_md5 = json.load(f)
            new_md5 = self.calc_all_md5()
            return new_md5 == old_md5
        return False

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return self._sample_shape

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Dogs and Cats dataset, shard number {self.rank} '
                f'out of {self.worldsize}')
