# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CIFAR10 Shard Descriptor (using `tf.data.Dataset` API)"""
import logging
from typing import List, Tuple

import tensorflow as tf

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class CIFAR10ShardDescriptor(ShardDescriptor):
    """
    CIFAR10 Shard Descriptor

    This example is based on `tf.data.Dataset` pipelines.
    Note that the ingestion of any model/task requires an iterable dataloader.
    Hence, it is possible to utilize these pipelines without explicit need of a
    new interface.
    """

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Download/Prepare CIFAR10 dataset"""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        # Load dataset
        train_ds, valid_ds = self._download_and_prepare_dataset(
            rank=self.rank,
            worldsize=self.worldsize
        )

        # Set attributes
        self._sample_shape = train_ds.element_spec[0].shape
        self._target_shape = train_ds.element_spec[1].shape

        self.splits = {
            'train': train_ds,
            'valid': valid_ds
        }

    @staticmethod
    def _download_and_prepare_dataset(rank: int, worldsize: int) -> Tuple[tf.data.Dataset]:
        """
        Load CIFAR10 as `tf.data.Dataset`.

        Provide `rank` and `worldsize` to auto-split uniquely for each client
        for simulation purposes.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Split
        x_train, y_train = x_train[rank - 1::worldsize], y_train[rank - 1::worldsize]
        x_test, y_test = x_test[rank - 1::worldsize], y_test[rank - 1::worldsize]

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        return train_ds, test_ds

    def get_shard_dataset_types(self) -> List[str]:
        """Get available split names"""
        return list(self.splits)

    def get_split(self, name: str) -> tf.data.Dataset:
        """Return a shard dataset by type."""
        if name not in self.splits:
            raise Exception(f'Split name `{name}` not found.'
                            f' Expected one of {list(self.splits.keys())}')
        return self.splits[name]

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return list(map(str, self._sample_shape))

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return list(map(str, self._target_shape))

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        n_train = len(self.splits['train'])
        n_test = len(self.splits['valid'])
        return (f'CIFAR10 dataset, shard number {self.rank}/{self.worldsize}.'
                f'\nSamples [Train/Valid]: [{n_train}/{n_test}]')
