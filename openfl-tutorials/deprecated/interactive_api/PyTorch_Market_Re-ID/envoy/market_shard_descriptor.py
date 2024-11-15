# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Market shard descriptor."""

import logging
import re
import zipfile
from pathlib import Path
from typing import List

import gdown
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class MarketShardDataset(ShardDataset):
    """Market shard dataset."""

    def __init__(self, dataset_dir: Path, dataset_type: str, rank=1, worldsize=1):
        """Initialize MarketShardDataset."""
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.rank = rank
        self.worldsize = worldsize

        self.imgs_path = list(dataset_dir.glob('*.jpg'))[self.rank - 1::self.worldsize]
        self.pattern = re.compile(r'([-\d]+)_c(\d)')

    def __len__(self):
        """Length of shard."""
        return len(self.imgs_path)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        img_path = self.imgs_path[index]
        pid, camid = map(int, self.pattern.search(img_path.name).groups())

        img = Image.open(img_path)
        return img, (pid, camid)


class MarketShardDescriptor(ShardDescriptor):
    """
    Market1501 Shard descriptor class.

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
        identities: 1501 (+1 for background)
        images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    def __init__(self, data_folder_name: str = 'Market-1501-v15.09.15',
                 rank_worldsize: str = '1,1') -> None:
        """Initialize MarketShardDescriptor."""
        super().__init__()

        # Settings for sharding the dataset
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.data_folder_name = data_folder_name
        self.dataset_dir = Path.cwd() / data_folder_name
        self.download()

        self.path_by_type = {
            'train': self.dataset_dir / 'bounding_box_train',
            'query': self.dataset_dir / 'query',
            'gallery': self.dataset_dir / 'bounding_box_test'
        }
        self._check_before_run()

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.path_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a dataset by type."""
        if dataset_type not in self.path_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}.'
                            f'Choose from the list: {", ".join(self.path_by_type)}')
        return MarketShardDataset(
            dataset_dir=self.path_by_type[dataset_type],
            dataset_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['64', '128', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['2']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Market dataset, shard number {self.rank} '
                f'out of {self.worldsize}')

    def download(self):
        """Download Market1501 dataset."""
        logger.info('Download Market1501 dataset.')
        if self.dataset_dir.exists():
            return None

        logger.info('Try to download.')
        output = f'{self.data_folder_name}.zip'

        if not Path(output).exists():
            url = 'https://drive.google.com/u/1/uc?id=0B8-rUzbwVRk0c054eEozWG9COHM'
            gdown.download(url, output, quiet=False)
        logger.info(f'{output} is downloaded.')

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(Path.cwd())

        Path(output).unlink()  # remove zip

    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        if not self.dataset_dir.exists():
            raise RuntimeError(f'{self.dataset_dir} does not exist')
        for dataset_path in self.path_by_type.values():
            if not dataset_path.exists():
                raise RuntimeError(f'{dataset_path} does not exist')
