# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Superb Shard Descriptor."""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from datasets import load_dataset
from datasets import load_metric

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class SuperbShardDataset(ShardDataset):
    """SUPERB Shard dataset class."""

    def __init__(self, dataset, rank: int = 1, worldsize: int = 1) -> None:
        """Initialize Superb shard Dataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.dataset = dataset
        self.x = self.dataset['audio'][self.rank - 1::self.worldsize]
        self.y = self.dataset['label'][self.rank - 1::self.worldsize]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return an item by the index."""
        return self.x[index]['array'], self.y[index]

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.x)


class SuperbShardDescriptor(ShardDescriptor):
    """Superb Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ) -> None:
        """Initialize SuperbShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        print('rank and worldsize', self.rank, self.worldsize)
        train_set, val_set, test_set = self.download_data()
        self.data_by_type = {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }

    def get_shard_dataset_types(self) -> List[Dict[str, Any]]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type: str = 'train') -> SuperbShardDataset:
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return SuperbShardDataset(
            self.data_by_type[dataset_type],
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return ['1']

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Superb dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def download_data(self) -> Tuple[Tuple[Dict, List], Tuple[Dict, List], Tuple[Dict, List]]:
        """Download dataset."""
        dataset = load_dataset('superb', 'ks')
        metric = load_metric('accuracy') # noqa

        # Train data
        train_set = dataset['train']

        # Validation data
        val_set = dataset['validation']

        # Test data
        test_set = dataset['test']

        print('Superb data was loaded!')
        return train_set, val_set, test_set
