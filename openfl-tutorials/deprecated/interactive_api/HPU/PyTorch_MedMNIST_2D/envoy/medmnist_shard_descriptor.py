# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MedMNIST Shard Descriptor."""

import logging
import os
from typing import Any, List, Tuple
from medmnist.info import INFO, HOMEPAGE

import numpy as np

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class MedMNISTShardDataset(ShardDataset):
    """MedMNIST Shard dataset class."""

    def __init__(self, x, y, data_type: str = 'train', rank: int = 1, worldsize: int = 1) -> None:
        """Initialize MedMNISTDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.x)


class MedMNISTShardDescriptor(ShardDescriptor):
    """MedMNIST Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            datapath: str = '',
            dataname: str = 'bloodmnist',
            **kwargs
    ) -> None:
        """Initialize MedMNISTShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.datapath = datapath
        self.dataset_name = dataname
        self.info = INFO[self.dataset_name]

        (x_train, y_train), (x_test, y_test) = self.load_data()
        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train') -> MedMNISTShardDataset:
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return MedMNISTShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return ['28', '28', '3']

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return ['1', '1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'MedMNIST dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    @staticmethod
    def download_data(datapath: str = 'data/',
                      dataname: str = 'bloodmnist',
                      info: dict = {}) -> None:

        logger.info(f"{datapath}\n{dataname}\n{info}")
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=info["url"],
                         root=datapath,
                         filename=dataname,
                         md5=info["MD5"])
        except Exception:
            raise RuntimeError('Something went wrong when downloading! '
                               + 'Go to the homepage to download manually. '
                               + HOMEPAGE)

    def load_data(self) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        """Download prepared dataset."""

        dataname = self.dataset_name + '.npz'
        dataset = os.path.join(self.datapath, dataname)

        if not os.path.isfile(dataset):
            logger.info(f"Dataset {dataname} not found at:{self.datapath}.\n\tDownloading...")
            MedMNISTShardDescriptor.download_data(self.datapath, dataname, self.info)
            logger.info("DONE!")

        data = np.load(dataset)

        x_train = data["train_images"]
        x_test = data["test_images"]

        y_train = data["train_labels"]
        y_test = data["test_labels"]
        logger.info('MedMNIST data was loaded!')
        return (x_train, y_train), (x_test, y_test)
