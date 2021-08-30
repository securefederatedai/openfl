# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Market shard descriptor."""

import re
import zipfile
from pathlib import Path

import gdown
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


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

    def __init__(self, datafolder: str = 'Market-1501-v15.09.15',
                 rank_worldsize: str = '1,1') -> None:
        """Initialize MarketShardDescriptor."""
        super().__init__()

        # Settings for sharding the dataset
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.download()
        self.pattern = re.compile(r'([-\d]+)_c(\d)')
        self.dataset_dir = Path.cwd() / datafolder
        self.train_dir = self.dataset_dir / 'bounding_box_train'
        self.query_dir = self.dataset_dir / 'query'
        self.gal_dir = self.dataset_dir / 'bounding_box_test'
        self._check_before_run()

        self.train_path = list(self.train_dir.glob('*.jpg'))[self.rank - 1::self.worldsize]
        self.query_path = list(self.query_dir.glob('*.jpg'))[self.rank - 1::self.worldsize]
        self.gal_path = list(self.gal_dir.glob('*.jpg'))[self.rank - 1::self.worldsize]

        self.mode = 'train'
        self.imgs_path = self.train_path

    def set_mode(self, mode='train'):
        """Set mode for getitem."""
        self.mode = mode
        if self.mode == 'train':
            self.imgs_path = self.train_path
        elif self.mode == 'query':
            self.imgs_path = self.query_path
        elif self.mode == 'gallery':
            self.imgs_path = self.gal_path
        else:
            raise Exception(f'Wrong mode: {mode}')

    def __len__(self):
        """Length of shard."""
        return len(self.imgs_path)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        img_path = self.imgs_path[index]
        pid, camid = map(int, self.pattern.search(img_path.name).groups())

        img = Image.open(img_path)
        return img, (pid, camid)

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

    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        if not self.dataset_dir.exists():
            raise RuntimeError(f'{self.dataset_dir} is not available')
        if not self.train_dir.exists():
            raise RuntimeError(f'{self.train_dir} is not available')
        if not self.query_dir.exists():
            raise RuntimeError(f'{self.query_dir} is not available')
        if not self.gal_dir.exists():
            raise RuntimeError(f'{self.gal_dir} is not available')

    @staticmethod
    def download():
        """Download Market1501 dataset."""
        zip_exists = False

        output = 'Market-1501-v15.09.15.zip'
        if Path(output).exists():
            zip_exists = True
            if Path('Market-1501-v15.09.15').exists():
                return None

        if not zip_exists:
            url = 'https://drive.google.com/uc?id=0B8-rUzbwVRk0c054eEozWG9COHM'
            gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(Path.cwd())

        Path(output).unlink()  # remove zip


if __name__ == '__main__':
    MarketShardDescriptor.download()
