# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MVTec shard descriptor."""

import os
from glob import glob
from pathlib import Path

import numpy as np
from imageio import imread
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class MVTecShardDataset(ShardDataset):
    """MVTec Shard dataset class."""

    def __init__(self, images_path,
                 mask_path, labels,
                 rank=1,
                 worldsize=1):
        """Initialize MVTecShardDataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.images_path = images_path[self.rank - 1::self.worldsize]
        self.mask_path = mask_path[self.rank - 1::self.worldsize]
        self.labels = labels[self.rank - 1::self.worldsize]

    def __getitem__(self, index):
        """Return a item by the index."""
        img = np.asarray(imread(self.images_path[index]))
        if img.shape[-1] != 3:
            img = self.gray2rgb(img)

        img = self.resize(img)
        img = np.asarray(img)
        label = self.labels[index]
        if self.mask_path[index]:
            mask = np.asarray(imread(self.mask_path[index]))
            mask = self.resize(mask)
            mask = np.asarray(mask)
        else:
            mask = np.zeros(img.shape)[:, :, 0]
        return img, mask, label

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.images_path)

    def resize(self, image, shape=(256, 256)):
        """Resize image."""
        return np.array(Image.fromarray(image).resize(shape))

    def gray2rgb(self, images):
        """Change image from gray to rgb."""
        tile_shape = tuple(np.ones(len(images.shape), dtype=int))
        tile_shape += (3,)

        images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
        return images


class MVTecShardDescriptor(ShardDescriptor):
    """MVTec Shard descriptor class."""

    def __init__(self, data_folder: str = 'MVTec_data',
                 rank_worldsize: str = '1,1',
                 obj: str = 'bottle'):
        """Initialize MVTecShardDescriptor."""
        super().__init__()

        self.dataset_path = Path.cwd() / data_folder
        self.download_data()
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        self.obj = obj

        # Calculating data and target shapes
        ds = self.get_dataset()
        sample, masks, target = ds[0]
        self._sample_shape = [str(dim) for dim in sample.shape]
        self._target_shape = [str(dim) for dim in target.shape]

    def download_data(self):
        """Download data."""
        zip_file_path = self.dataset_path / 'mvtec_anomaly_detection.tar.xz'
        if not Path(zip_file_path).exists():
            os.makedirs(self.dataset_path, exist_ok=True)
            print('Downloading MVTec Dataset...this might take a while')
            os.system('wget -nc'
                      " 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'"  # noqa
                      f' -O {zip_file_path.relative_to(Path.cwd())}')
            print('Downloaded MVTec dataset, untar-ring now')
            os.system(f'tar -xvf {zip_file_path.relative_to(Path.cwd())}'
                      f' -C {self.dataset_path.relative_to(Path.cwd())}')
            # change to write permissions
            self.change_permissions(self.dataset_path, 0o764)

    def change_permissions(self, folder, code):
        """Change permissions after data is downloaded."""
        for root, dirs, files in os.walk(folder):
            for d in dirs:
                os.chmod(os.path.join(root, d), code)
            for f in files:
                os.chmod(os.path.join(root, f), code)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        # Train dataset
        if dataset_type == 'train':
            fpattern = os.path.join(self.dataset_path, f'{self.obj}/train/*/*.png')
            fpaths = sorted(glob(fpattern))
            self.images_path = list(fpaths)
            self.labels = np.zeros(len(fpaths), dtype=np.int32)
            # Masks
            self.mask_path = np.full(self.labels.shape, None)
        # Test dataset
        elif dataset_type == 'test':
            fpattern = os.path.join(self.dataset_path, f'{self.obj}/test/*/*.png')
            fpaths = sorted(glob(fpattern))
            fpaths_anom = list(
                filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
            fpaths_good = list(
                filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))
            fpaths = fpaths_anom + fpaths_good
            self.images_path = fpaths
            self.labels = np.zeros(len(fpaths_anom) + len(fpaths_good), dtype=np.int32)
            self.labels[:len(fpaths_anom)] = 1   # anomalies
            # Masks
            fpattern_mask = os.path.join(self.dataset_path, f'{self.obj}/ground_truth/*/*.png')
            self.mask_path = sorted(glob(fpattern_mask)) + [None] * len(fpaths_good)
        else:
            raise Exception(f'Wrong dataset type: {dataset_type}.'
                            f'Choose from the list: [train, test]')

        return MVTecShardDataset(
            images_path=self.images_path,
            mask_path=self.mask_path,
            labels=self.labels,
            rank=self.rank,
            worldsize=self.worldsize,
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['256', '256', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['256', '256']

    @property
    def dataset_description(self) -> str:
        """Return the shard dataset description."""
        return (f'MVTec dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
