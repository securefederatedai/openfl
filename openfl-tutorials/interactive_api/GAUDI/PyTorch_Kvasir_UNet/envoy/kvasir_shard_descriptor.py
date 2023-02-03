# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Kvasir shard descriptor."""

import os
from pathlib import Path

import numpy as np
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.utilities import validate_file_hash


class KvasirShardDataset(ShardDataset):
    """Kvasir Shard dataset class."""

    def __init__(self, dataset_dir: Path, rank=1, worldsize=1, enforce_image_hw=None):
        """Initialize KvasirShardDataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.dataset_dir = dataset_dir
        self.enforce_image_hw = enforce_image_hw
        self.images_path = self.dataset_dir / 'segmented-images' / 'images'
        self.masks_path = self.dataset_dir / 'segmented-images' / 'masks'

        self.images_names = [
            img_name
            for img_name in sorted(os.listdir(self.images_path))
            if len(img_name) > 3 and img_name[-3:] == 'jpg'
        ]
        # Sharding
        self.images_names = self.images_names[self.rank - 1::self.worldsize]

    def __getitem__(self, index):
        """Return a item by the index."""
        name = self.images_names[index]
        # Reading data
        img = Image.open(self.images_path / name)
        mask = Image.open(self.masks_path / name)
        if self.enforce_image_hw is not None:
            # If we need to resize data
            # PIL accepts (w,h) tuple, not (h,w)
            img = img.resize(self.enforce_image_hw[::-1])
            mask = mask.resize(self.enforce_image_hw[::-1])
        img = np.asarray(img)
        mask = np.asarray(mask)
        assert img.shape[2] == 3

        return img, mask[:, :, 0].astype(np.uint8)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.images_names)


class KvasirShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, data_folder: str = 'kvasir_data',
                 rank_worldsize: str = '1,1',
                 enforce_image_hw: str = None) -> None:
        """Initialize KvasirShardDescriptor."""
        super().__init__()
        # Settings for sharding the dataset
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.data_folder = Path.cwd() / data_folder
        self.download_data(self.data_folder)

        # Settings for resizing data
        self.enforce_image_hw = None
        if enforce_image_hw is not None:
            self.enforce_image_hw = tuple(int(size) for size in enforce_image_hw.split(','))

        # Calculating data and target shapes
        ds = self.get_dataset()
        sample, target = ds[0]
        self._sample_shape = [str(dim) for dim in sample.shape]
        self._target_shape = [str(dim) for dim in target.shape]

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        return KvasirShardDataset(
            dataset_dir=self.data_folder,
            rank=self.rank,
            worldsize=self.worldsize,
            enforce_image_hw=self.enforce_image_hw
        )

    @staticmethod
    def download_data(data_folder):
        """Download data."""
        zip_file_path = data_folder / 'kvasir.zip'
        os.makedirs(data_folder, exist_ok=True)
        os.system(
            'wget -nc'
            " 'https://datasets.simula.no/downloads/"
            "hyper-kvasir/hyper-kvasir-segmented-images.zip'"
            f' -O {zip_file_path.relative_to(Path.cwd())}'
        )
        zip_sha384 = ('66cd659d0e8afd8c83408174'
                      '1ade2b75dada8d4648b816f2533c8748b1658efa3d49e205415d4116faade2c5810e241e')
        validate_file_hash(zip_file_path, zip_sha384)
        os.system(f'unzip -n {zip_file_path.relative_to(Path.cwd())}'
                  f' -d {data_folder.relative_to(Path.cwd())}')

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return self._sample_shape

    @property
    def target_shape(self):
        """Return the target shape info."""
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Kvasir dataset, shard number {self.rank} '
                f'out of {self.worldsize}')


if __name__ == '__main__':
    from openfl.interface.cli import setup_logging

    setup_logging()

    data_folder = 'data'
    rank_worldsize = '1,100'
    enforce_image_hw = '529,622'

    kvasir_sd = KvasirShardDescriptor(
        data_folder,
        rank_worldsize=rank_worldsize,
        enforce_image_hw=enforce_image_hw)

    print(kvasir_sd.dataset_description)
    print(kvasir_sd.sample_shape, kvasir_sd.target_shape)

    from openfl.component.envoy.envoy import Envoy

    shard_name = 'one'
    director_host = 'localhost'
    director_port = 50051

    keeper = Envoy(
        shard_name=shard_name,
        director_host=director_host,
        director_port=director_port,
        shard_descriptor=kvasir_sd,
        tls=True,
        root_certificate='./cert/root_ca.crt',
        private_key='./cert/one.key',
        certificate='./cert/one.crt',
    )

    keeper.start()
