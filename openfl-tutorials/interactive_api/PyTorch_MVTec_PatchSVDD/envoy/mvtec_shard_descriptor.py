# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MVTec shard descriptor."""

import os
from glob import glob

from pathlib import Path
import numpy as np
from PIL import Image
from imageio import imread

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class MVTecShardDescriptor(ShardDescriptor):
    """MVTec Shard descriptor class."""

    def __init__(self, data_folder: str = 'MVTec_data',
                 rank_worldsize: str = '1,1',
                 enforce_image_hw: str = None,
                 obj: str = '',
                 mode: str = 'train') -> None:
        """Initialize MVTecShardDescriptor."""
        super().__init__()

        self.data_folder = Path.cwd() / data_folder
        self.download_data(self.data_folder)
        self.dataset_path = self.data_folder
        # Settings for resizing data
        self.enforce_image_hw = None
        if enforce_image_hw is not None:
            self.enforce_image_hw = tuple(int(size) for size in enforce_image_hw.split(','))
        # Settings for sharding the dataset
        self.rank_worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        self.mode = mode
        # Train dataset
        fpattern = os.path.join(self.dataset_path, f'{obj}/train/*/*.png')
        fpaths = sorted(glob(fpattern))
        self.train_path = list(fpaths)[self.rank_worldsize[0] - 1::self.rank_worldsize[1]]
        # Test dataset
        fpattern = os.path.join(self.dataset_path, f'{obj}/test/*/*.png')
        fpaths = sorted(glob(fpattern))
        fpaths1 = list(
            filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
        fpaths2 = list(
            filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))
        fpaths = fpaths1 + fpaths2
        self.test_path = fpaths[self.rank_worldsize[0] - 1::self.rank_worldsize[1]]
        # Sharding the labels
        self.labels = np.zeros(len(fpaths1) + len(fpaths2), dtype=np.int32)
        self.labels[:len(fpaths1)] = 1   # anomalies
        self.labels = self.labels[self.rank_worldsize[0] - 1::self.rank_worldsize[1]]
        # Masks
        fpattern_mask = os.path.join(self.dataset_path, f'{obj}/ground_truth/*/*.png')
        self.fpaths_mask = sorted(glob(fpattern_mask))
        self.len_anomaly = len(self.fpaths_mask)
        self.mask_path = self.fpaths_mask

    def set_mode(self, mode='train'):
        """Set mode for getitem."""
        self.mode = mode
        if self.mode == 'train':
            self.imgs_path = self.train_path
        elif self.mode == 'test':
            self.imgs_path = self.test_path
        else:
            raise Exception(f'Wrong mode: {mode}')

    @staticmethod
    def download_data(data_folder):
        """Download data."""
        zip_exists = False
        zip_file_path = data_folder / 'mvtec_anomaly_detection.tar.xz'
        if Path(zip_file_path).exists():
            zip_exists = True

        if not zip_exists:
            os.makedirs(data_folder, exist_ok=True)
            print('Downloading MVTec Dataset...this might take a while')
            os.system('wget -nc'
                      " 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'"  # noqa
                      f' -O {zip_file_path.relative_to(Path.cwd())}')
            print('Downloaded MVTec dataset, untar-ring now')
            os.system(f'tar -xvf {zip_file_path.relative_to(Path.cwd())}'
                      f' -C {data_folder.relative_to(Path.cwd())}')

    def __getitem__(self, index):
        """Return a item by the index."""
        img = np.asarray(imread(self.imgs_path[index]))
        if img.shape[-1] != 3:
            img = self.gray2rgb(img)

        img = self.resize(img)
        img = np.asarray(img)
        if self.mode == 'train':
            mask = np.full(img.shape, None)
            label = 0
        else:
            if self.mask_path[index]:
                mask = np.asarray(imread(self.mask_path[index]))
                mask = self.resize(mask)
                mask = np.asarray(mask)
                label = self.labels[index]
            else:
                mask = np.full(img.shape, None)
                label = self.labels[index]

        return img, mask, label

    def resize(self, image, shape=(256, 256)):
        """Resize image."""
        return np.array(Image.fromarray(image).resize(shape))

    def gray2rgb(self, images):
        """Change image from gray to rgb."""
        tile_shape = tuple(np.ones(len(images.shape), dtype=int))
        tile_shape += (3,)

        images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
        return images

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.train_path)

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
        """Return the dataset description."""
        return (f'MVTec dataset, shard number {self.rank_worldsize[0]}'
                f' out of {self.rank_worldsize[1]}')
