# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Market shard descriptor."""

import re
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


logger = getLogger(__name__)

DATAPATH = list(Path.cwd().parents[2].rglob('**/Market'))[0]    # parent directory of project


class MarketShardDescriptor(ShardDescriptor):
    """
    Market1501 Shard descriptor class.

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    Download data: https://www.kaggle.com/pengcw1/market-1501

    Dataset statistics:
        identities: 1501 (+1 for background)
        images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    def __init__(self, rank_worldsize: str = '1,1') -> None:
        """Initialize MarketShardDescriptor."""
        super().__init__()

        # Settings for sharding the dataset
        self.rank_worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.pattern = re.compile(r'([-\d]+)_c(\d)')
        self.dataset_dir = Path(DATAPATH)
        self.train_dir = self.dataset_dir / 'bounding_box_train'
        self.query_dir = self.dataset_dir / 'query'
        self.gallery_dir = self.dataset_dir / 'bounding_box_test'

        self._check_before_run()

        self.train, self.num_train_pids, self.num_train_imgs = self._process_dir(
            self.train_dir, relabel=True
        )
        self.query, self.num_query_pids, self.num_query_imgs = self._process_dir(
            self.query_dir, relabel=False
        )
        self.gallery, self.num_gallery_pids, self.num_gallery_imgs = self._process_dir(
            self.gallery_dir, relabel=False
        )

        num_total_pids = self.num_train_pids + self.num_query_pids
        num_total_imgs = self.num_train_imgs + self.num_query_imgs + self.num_gallery_imgs

        logger.info(
            '=> Market1501 loaded\n'
            'Dataset statistics:\n'
            '  ------------------------------\n'
            '  subset   | # ids | # images\n'
            '  ------------------------------\n'
            f'  train    | {self.num_train_pids} | {self.num_train_imgs}\n'
            f'  query    | {self.num_query_pids} | {self.num_query_imgs}\n'
            f'  gallery  | {self.num_gallery_pids} | {self.num_gallery_imgs}\n'
            '------------------------------\n'
            f'total    | {num_total_pids} | {num_total_imgs}\n'
            '  ------------------------------'
        )

    def __len__(self):
        """Length of shard."""
        start = self.rank_worldsize[0] - 1
        step = self.rank_worldsize[1]
        return len(list(self.train_dir.glob('*.jpg'))[start::step])

    def __getitem__(self, index: int):
        """Return a item by the index."""
        start = self.rank_worldsize[0] - 1
        step = self.rank_worldsize[1]
        imgs_path = list(self.train_dir.glob('*.jpg'))[start::step]
        img_path = imgs_path[index]
        pid, _ = map(int, self.pattern.search(img_path.name).groups())

        img = Image.open(img_path)
        img = np.asarray(img)
        return img, pid

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['64', '128', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1501']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Market dataset, shard number {self.rank_worldsize[0]} '
                f'out of {self.rank_worldsize[1]}')

    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        if not self.dataset_dir.exists():
            raise RuntimeError(f'{self.dataset_dir} is not available')
        if not self.train_dir.exists():
            raise RuntimeError(f'{self.train_dir} is not available')
        if not self.query_dir.exists():
            raise RuntimeError(f'{self.query_dir} is not available')
        if not self.gallery_dir.exists():
            raise RuntimeError(f'{self.gallery_dir} is not available')

    def _process_dir(self, dir_path, relabel=False, label_start=0):
        """Get data from directory."""
        pattern = re.compile(r'([-\d]+)_c(\d)')
        start = self.rank_worldsize[0] - 1
        step = self.rank_worldsize[1]
        img_paths = list(dir_path.glob('*.jpg'))[start::step]

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path.name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path.name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            if label_start == 0:
                assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid] + label_start
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
