# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Noisy-Sin Shard Descriptor."""

from typing import List, Dict
from pathlib import Path
import requests

import numpy as np

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.utilities import validate_file_hash
from openfl.utilities.data_splitters import EqualNumPyDataSplitter

class MedMnistShardDataset(ShardDataset):
    """Dogs and cats Shard dataset class."""

    def __init__(self, data_type: str, data: Dict[str, np.ndarray],
                 rank: int = 1, worldsize: int = 1):
        """Initialize MedMnistShardDataset."""
        
        if data_type == 'train':
            self.imgs = data['train_images']
            self.labels = data['train_labels']
        elif data_type == 'val':
            self.imgs = data['val_images']
            self.labels = data['val_labels']
        
        # Sharding
        self.data_splitter = EqualNumPyDataSplitter()
        train_idx = data_splitter.split(self.labels, worldsize)[rank]
        valid_idx = data_splitter.split(self.labels, worldsize)[rank]

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)

        img = img/255.

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.imgs)

class MedMnistSD(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, root = './data', rank: int = 1, worldsize: int = 1) -> None:
        """
        Initialize LinReg Shard Descriptor.

        This Shard Descriptor generate random data. Sample features are
        floats between pi/3 and 5*pi/3, and targets are calculated
        calculated as sin(feature) + normal_noise.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(exist_ok=True)
        local_file_path = self.root / 'organmnist3d.npz'
        url = 'https://zenodo.org/record/5208230/files/organmnist3d.npz?download=1'
        md5 = '21f0a239e7f502e6eca33c3fc453c0b6'
        response = requests.get(url)
        with open(local_file_path, 'wb') as f:
            f.write(response.content)
        validate_file_hash(local_file_path, md5)
        npz_file = np.load(local_file_path)
        
        
        train_data = MedMnistShardDataset('train', npz_file, rank, worldsize)
        valid_data = MedMnistShardDataset('val', npz_file, rank, worldsize)
        
        self.train_data = train_data
        self.train_data.imgs = train_data.imgs[train_idx]
        self.train_data.labels = train_data.labels[train_idx]
        self.valid_data = valid_data
        self.valid_data.imgs = valid_data.imgs[valid_idx]
        self.valid_data.labels = valid_data.labels[valid_idx]
        
        

    def get_dataset(self, dataset_type: str) -> np.ndarray:
        """
        Return a shard dataset by type.

        A simple list with elements (x, y) implemets the Shard Dataset interface.
        """
        if dataset_type == 'train':
            return self.train_data
        elif dataset_type == 'val':
            return self.valid_data
        else:
            pass

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        (*x, _) = self.train_data[0]
        return [str(i) for i in np.array(x, ndmin=1).shape]

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        (*_, y) = self.train_data[0]
        return [str(i) for i in np.array(y, ndmin=1).shape]

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return 'Allowed dataset types are `train` and `val`'
