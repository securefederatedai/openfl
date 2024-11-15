# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from torchvision import datasets

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class MnistShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank: int = 1, worldsize: int = 1) -> None:
        """Initialize Mnist shard Dataset."""
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


class MnistShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ) -> None:
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        (x_train, y_train), (x_val, y_val) = self.download_data()
        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_val, y_val)
        }

    def get_shard_dataset_types(self) -> List[Dict[str, Any]]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type: str = 'train') -> MnistShardDataset:
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return MnistShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return ['28', '28']

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Mnist dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def download_data(self) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        """Download prepared dataset."""
        train_data, val_data = (
            datasets.MNIST('data', train=train, download=True)
            for train in (True, False)
        )
        x_train, y_train = train_data.train_data, train_data.train_labels
        x_val, y_val = val_data.test_data, val_data.test_labels

        print('Mnist data was loaded!')
        return (x_train, y_train), (x_val, y_val)
