# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CIFAR10 Shard Descriptor (using `TFDS` API)"""
import jax.numpy as jnp
import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import List, Tuple
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class CIFAR10ShardDescriptor(ShardDescriptor):
    """
    CIFAR10 Shard Descriptor

    This example is based on `tfds` data loader.
    Note that the ingestion of any model/task requires an iterable dataloader.
    Hence, it is possible to utilize these pipelines without explicit need of a
    new interface.
    """

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ) -> None:
        """Download/Prepare CIFAR10 dataset"""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        # Load dataset
        train_ds, valid_ds = self._download_and_prepare_dataset(self.rank, self.worldsize)

        # Set attributes
        self._sample_shape = train_ds['image'].shape[1:]
        self._target_shape = tf.expand_dims(train_ds['label'], -1).shape[1:]

        self.splits = {
            'train': train_ds,
            'valid': valid_ds
        }

    def _download_and_prepare_dataset(self, rank: int, worldsize: int) -> Tuple[dict]:
        """
        Download, Cache CIFAR10 and prepare `tfds` builder.

        Provide `rank` and `worldsize` to virtually split dataset across shards
        uniquely for each client for simulation purposes.

        Returns:
        Tuple (train_dict, test_dict) of dictionary with JAX DeviceArray (image and label)
        dict['image'] -> DeviceArray float32
        dict['label'] -> DeviceArray int32
        {'image' : DeviceArray(...), 'label' : DeviceArray(...)}

        """

        dataset_builder = tfds.builder('cifar10')
        dataset_builder.download_and_prepare()

        datasets = dataset_builder.as_dataset()

        train_shard_size = int(len(datasets['train']) / worldsize)
        test_shard_size = int(len(datasets['test']) / worldsize)

        self.train_segment = f'train[{train_shard_size * (rank - 1)}:{train_shard_size * rank}]'
        self.test_segment = f'test[{test_shard_size * (rank - 1)}:{test_shard_size * rank}]'
        train_dataset = dataset_builder.as_dataset(split=self.train_segment, batch_size=-1)
        test_dataset = dataset_builder.as_dataset(split=self.test_segment, batch_size=-1)
        train_ds = tfds.as_numpy(train_dataset)
        test_ds = tfds.as_numpy(test_dataset)

        train_ds['image'] = jnp.float32(train_ds['image']) / 255.
        test_ds['image'] = jnp.float32(test_ds['image']) / 255.
        train_ds['label'] = jnp.int32(train_ds['label'])
        test_ds['label'] = jnp.int32(test_ds['label'])

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
        n_train = len(self.splits['train']['label'])
        n_test = len(self.splits['valid']['label'])
        return (f'CIFAR10 dataset, Shard Segments {self.train_segment}/{self.test_segment}, '
                f'rank/world {self.rank}/{self.worldsize}.'
                f'\n num_samples [Train/Valid]: [{n_train}/{n_test}]')
