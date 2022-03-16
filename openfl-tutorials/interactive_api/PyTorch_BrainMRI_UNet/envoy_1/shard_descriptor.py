# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""BrainMRI Shard Descriptor"""

import logging
from typing import Any, Dict, List, Tuple
import os
import numpy as np
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)

class brainMRIShard(ShardDataset):
    """brainMRI Shard dataset class."""

    def __init__(self, csv_path, rank=1, worldsize=1,
                crop_dims=None, dataset_dir=None,
                augment=True, device='cpu', set='train') -> None:
        """Initialize brainMRIShardDataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.csv = pd.read_csv(csv_path)
        self.crop_dim = crop_dims
        self.dataset_dir = dataset_dir
        # Shard Dataset
        if set=='train':
            if self.rank==self.worldsize:
                self.csv = self.csv.loc[(self.rank-1)*(len(self.csv)//self.worldsize):,:].reset_index(drop=True)
            else:
                self.csv = self.csv.loc[(self.rank-1)*(len(self.csv)//self.worldsize):(self.rank)*(len(self.csv)//self.worldsize),:].reset_index(drop=True)
        self.device = device
        self.augment = augment


    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """Return a item by the index."""
        img_path = os.path.join(self.dataset_dir,self.csv.loc[idx,'img_path'])
        msk_path = os.path.join(self.dataset_dir,self.csv.loc[idx,'msk_path'])
        img = np.load(img_path)
        msk = np.load(msk_path)
        if self.crop_dim is None:
            self.crop_dim = [img.shape[0],img.shape[1]]
        img = self.preprocess_img(img)
        msk = self.preprocess_label(msk)
        img,msk = self.crop_input(img,msk)
        if self.augment:
            img,msk = self.augment_data(img,msk)
        img = img[np.newaxis,:,:].copy()
        msk = msk.copy()
        img = torch.tensor(img).to(self.device).float()
        msk = torch.tensor(msk).to(self.device).int()
        return img,msk


    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.csv)


    def crop_input(self, img, msk) -> Tuple[Any, Any]:
        """
        Randomly crop the image and mask
        """
        # If crop_dim == -1, then don't crop
        if self.crop_dim[0] == -1:
            self.crop_dim[0] = img.shape[0]
        if self.crop_dim[1] == -1:
            self.crop_dim[1] = img.shape[1]  

        slices = []

        # Do we randomize?
        is_random = self.augment and np.random.rand() > 0.5

        for idx, idy in enumerate(range(2)):  # Go through each dimension

            cropLen = self.crop_dim[idx]
            imgLen = img.shape[idy]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):  # Don't fall off the image
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk) -> Tuple[Any, Any]:
        if np.random.rand() > 0.5:
            ax = np.random.choice([0,1])
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            img = np.rot90(img, rot, axes=[0,1])  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=[0,1])  # Rotate axes 0 and 1

        return img, msk

    def preprocess_img(self, img) -> Any:
        EPSILON = 1e-8
        return (img - img.mean()) / (img.std() + EPSILON)
    

    def preprocess_label(self, label) -> Any:
        label[label> 0] = 1.0
        return label



class brainMRIShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, train_csv_path: str, valid_csv_path: str, rank_worldsize: str = '1,1',
                 crop_dims: str = None) -> None:
        """Initialize brainMRIShardDescriptor."""
        super().__init__()

        # CSVs to train and test
        self.train_csv = train_csv_path
        self.valid_csv = valid_csv_path

        # Settings for sharding the dataset
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        # Calculating data and target shapes
        if crop_dims is not None:
            self.crop_dims = [crop_dims, crop_dims]
        else:
            self.crop_dims = None

        ds = self.get_dataset('train')
        sample, target = ds[0]
        self._sample_shape = [str(dim) for dim in sample.shape]
        self._target_shape = [str(dim) for dim in target.shape]


    def get_dataset(self, dataset_type='train') -> brainMRIShard:
        # """Return a shard dataset by type."""
        if dataset_type=='train':
            return brainMRIShard(
                csv_path=self.train_csv,
                rank=self.rank,
                worldsize=self.worldsize,
                crop_dims=self.crop_dims
            )
        elif dataset_type=='valid':
            return brainMRIShard(
                csv_path=self.valid_csv,
                rank=-1,
                worldsize=-1,
                crop_dims=self.crop_dims,
                augment=False,
                set='valid'
            )    
    
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
        return (f'brainMRI dataset, shard number {self.rank} '
                f'out of {self.worldsize}')