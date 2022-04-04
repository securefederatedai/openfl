# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Noisy-Sin Shard Descriptor."""

from typing import List, Dict
from pathlib import Path
import requests
import hashlib
import tqdm

import numpy as np

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.utilities.data_splitters import EqualNumPyDataSplitter

class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
   
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        return voxel.astype(np.float32)

def validate_md5(file_path, expected_hash, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(file_path, 'rb') as file:
        # Reading is buffered, so we can read smaller chunks.
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    if h.hexdigest() != expected_hash:
        raise SystemError('ZIP File hash doesn\'t match expected file hash.')

class MedMnistShardDataset(ShardDataset):
    """MedMNIST Shard dataset class."""

    def __init__(self, data_type: str, data: Dict[str, np.ndarray],
                 rank: int = 1, worldsize: int = 1, transform=None, target_transform=None):
        """Initialize MedMnistShardDataset."""
        
        if data_type == 'train':
            self.imgs = data['train_images']
            self.labels = data['train_labels']
        elif data_type == 'val':
            self.imgs = data['val_images']
            self.labels = data['val_labels']
        
        # Sharding
        num_classes = len(np.unique(self.labels))
        
        seed=0
        data_splitter = EqualNumPyDataSplitter()
        split_idx = data_splitter.split(self.labels, worldsize)
        while any([len(np.unique(self.labels[i])) != num_classes for i in split_idx]):
            # we need all classes to be on all envoys to compute roc auc metric
            seed += 1
            # seed = np.random.randint(seed)
            data_splitter = EqualNumPyDataSplitter(seed=seed)
            split_idx = data_splitter.split(self.labels, worldsize)
        split_idx = split_idx[rank]
        self.imgs = self.imgs[split_idx]
        self.labels = self.labels[split_idx]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]

        img = np.stack([img/255.]*1, axis=0)

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
        Initialize MedMNIST Shard Descriptor.

        This Shard Descriptor generate random data. Sample features are
        floats between pi/3 and 5*pi/3, and targets are calculated 
        calculated as sin(feature) + normal_noise.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(exist_ok=True)
        local_file_path = self.root / 'organmnist3d.npz'
        md5 = '21f0a239e7f502e6eca33c3fc453c0b6'
        if local_file_path.exists():
            validate_md5(local_file_path, md5)
        else:
            url = 'https://zenodo.org/record/5208230/files/organmnist3d.npz?download=1'
            chunk_size = 1024
            response = requests.get(url, stream=True)
            with open(local_file_path, 'wb') as f:
                with tqdm(total=response.length) as pbar:
                    for chunk in iter(lambda: response.iter_content(chunk_size), ""):
                        pbar.update(chunk_size)
                        f.write(chunk)
            validate_md5(local_file_path, md5)
        npz_file = np.load(local_file_path)
        
        self.train_data = MedMnistShardDataset('train', npz_file, rank, worldsize, transform=Transform3D())
        self.valid_data = MedMnistShardDataset('val', npz_file, rank, worldsize, transform=Transform3D())
        
        

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
        img, _ = self.train_data[0]
        return [str(i) for i in img.shape]

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        _, label = self.train_data[0]
        return [str(i) for i in label.shape]

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return 'Allowed dataset types are `train` and `val`'
