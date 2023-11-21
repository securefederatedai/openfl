# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""BrainMRI Shard Descriptor"""

import logging
from typing import Any, Tuple
import os
import numpy as np
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
import pandas as pd
import torch
import tarfile
import gdown
import nibabel as nib
from tqdm import tqdm
import sys
import json


logger = logging.getLogger(__name__)


class brainMRIShard(ShardDataset):
    """brainMRI Shard dataset class."""

    def __init__(self, csv_path, rank=1, worldsize=1,
                 crop_dims=None, dataset_dir='./',
                 augment=True, device='cpu', set='train') -> None:
        """Initialize brainMRIShardDataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.csv = pd.read_csv(csv_path)
        self.crop_dim = crop_dims
        self.dataset_dir = dataset_dir
        # Shard Dataset
        if set == 'train':
            if self.rank == self.worldsize:
                self.csv = self.csv.loc[(self.rank - 1) * (
                    len(self.csv) // self.worldsize):, :].reset_index(
                        drop=True)
            else:
                self.csv = self.csv.loc[(self.rank - 1) * (len(
                    self.csv) // self.worldsize):(self.rank) * (len(
                        self.csv) // self.worldsize), :].reset_index(drop=True)
        self.device = device
        self.augment = augment

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """Return a item by the index."""
        img_path = os.path.join(self.dataset_dir, self.csv.loc[idx, 'img_path'])
        msk_path = os.path.join(self.dataset_dir, self.csv.loc[idx, 'msk_path'])
        img = np.load(img_path)
        msk = np.load(msk_path)
        if self.crop_dim is None:
            self.crop_dim = [img.shape[0], img.shape[1]]
        img = self.preprocess_img(img)
        msk = self.preprocess_label(msk)
        img, msk = self.crop_input(img, msk)
        if self.augment:
            img, msk = self.augment_data(img, msk)
        img = img[np.newaxis, :, :].copy()
        msk = msk.copy()
        img = torch.tensor(img).to(self.device).float()
        msk = torch.tensor(msk).to(self.device).int()
        return img, msk

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

            start = (imgLen - cropLen) // 2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start * ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):  # Don't fall off the image
                        start = (imgLen - cropLen) // 2
            else:
                start = 0

            slices.append(slice(start, start + cropLen))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk) -> Tuple[Any, Any]:
        if np.random.rand() > 0.5:
            ax = np.random.choice([0, 1])
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            img = np.rot90(img, rot, axes=[0, 1])  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=[0, 1])  # Rotate axes 0 and 1

        return img, msk

    def preprocess_img(self, img) -> Any:
        EPSILON = 1e-8
        return (img - img.mean()) / (img.std() + EPSILON)

    def preprocess_label(self, label) -> Any:
        label[label > 0] = 1.0
        return label


class brainMRIShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, train_csv_path: str, valid_csv_path: str,
                 rank_worldsize: str = '1,1', crop_dims: str = None) -> None:
        """Initialize brainMRIShardDescriptor."""
        super().__init__()

        self.download_dataset()

        # CSVs to train and test
        self.train_csv = train_csv_path
        self.valid_csv = valid_csv_path

        # Settings for sharding the dataset
        self.rank, self.worldsize = tuple(int(
            num) for num in rank_worldsize.split(','))

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
        if dataset_type == 'train':
            return brainMRIShard(
                csv_path=self.train_csv,
                rank=self.rank,
                worldsize=self.worldsize,
                crop_dims=self.crop_dims
            )
        elif dataset_type == 'valid':
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

    @staticmethod
    def download_dataset() -> None:
        """Helper function to download the dataset"""
        # If already exists
        if os.path.exists('Task01_BrainTumour.tar'):
            return None
        url = 'https://drive.google.com/uc?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU'
        gdown.download(url, output='Task01_BrainTumour.tar', quiet=False)
        # use tarfile to extract the data
        tar = tarfile.open('./Task01_BrainTumour.tar')
        tar.extractall()
        tar.close()
        # Now read the data and process it to make the necessary images in
        # the way we want it to be
        data_folder = './Task01_BrainTumour/'
        dataset_json_path = data_folder + 'dataset.json'
        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)
        image_list = dataset_json['training']
        num_images = len(image_list)
        num_train = int(np.floor(num_images * .80))
        train_image_list = image_list[:num_train]
        valid_image_list = image_list[num_train:]
        # The processed images are saved in ./processed_data/ folder
        save_folder = './processed_data/'
        if os.path.exists(save_folder):
            print("processed_data/ already exists, do you want to proceed? \
                press y/Y to proceed and n/N to cancel")
            inp = input()
            if inp == 'y' or inp == 'Y':
                pass
            else:
                sys.exit("Exiting program")
        else:
            os.mkdir(save_folder)

        img_path_new = save_folder + 'img/'
        if os.path.exists(img_path_new):
            print("processed_data/img/ already exists, \
                do you want to proceed? press y/Y to proceed and n/N to cancel")
            inp = input()
            if inp == 'y' or inp == 'Y':
                pass
            else:
                sys.exit("Exiting program")
        else:
            os.mkdir(img_path_new)

        msk_path_new = save_folder + 'msk/'
        if os.path.exists(msk_path_new):
            print("processed_data/msk/ already exists, do you \
                want to proceed? press y/Y to proceed and n/N to cancel")
            inp = input()
            if inp == 'y' or inp == 'Y':
                pass
            else:
                sys.exit("Exiting program")
        else:
            os.mkdir(msk_path_new)

        # Two CSV files 'train_data.csv' and 'valid_data.csv' are created which
        #  will have the maps
        train_data_csv = './train_data.csv'
        valid_data_csv = './valid_data.csv'
        with open(train_data_csv, 'w') as f:
            f.write('name,img_path,msk_path\n')
        with open(valid_data_csv, 'w') as f:
            f.write('name,img_path,msk_path\n')

        # process the train images
        for t in tqdm(train_image_list):
            img_name = t['image'][t['image'].rfind('/') + 1:t['image'].rfind('.nii')]
            img_path = os.path.join(data_folder, t['image'])
            msk_path = os.path.join(data_folder, t['label'])
            img = np.array(nib.load(img_path).dataobj)[:, :, :, 0]
            msk = np.array(nib.load(msk_path).dataobj)
            num_slices = msk.shape[2]
            for slice in range(num_slices):
                img_slice = img[:, :, slice]
                msk_slice = msk[:, :, slice]
                img_slice_name = os.path.join(img_path_new, f'{img_name}_{slice}.npy')
                msk_slice_name = os.path.join(msk_path_new, f'{img_name}_{slice}.npy')
                np.save(img_slice_name, img_slice)
                np.save(msk_slice_name, msk_slice)
                with open(train_data_csv, 'a') as f:
                    f.write(f'{img_name},{img_slice_name},{msk_slice_name}\n')

        # process the validation images
        for v in tqdm(valid_image_list):
            img_name = v['image'][v['image'].rfind('/') + 1:v['image'].rfind('.nii')]
            img_path = os.path.join(data_folder, v['image'])
            msk_path = os.path.join(data_folder, v['label'])
            img = np.array(nib.load(img_path).dataobj)[:, :, :, 0]
            msk = np.array(nib.load(msk_path).dataobj)
            num_slices = msk.shape[2]
            for slice in range(num_slices):
                img_slice = img[:, :, slice]
                msk_slice = msk[:, :, slice]
                img_slice_name = os.path.join(img_path_new, f'{img_name}_{slice}.npy')
                msk_slice_name = os.path.join(msk_path_new, f'{img_name}_{slice}.npy')
                np.save(img_slice_name, img_slice)
                np.save(msk_slice_name, msk_slice)
                with open(valid_data_csv, 'a') as f:
                    f.write(f'{img_name},{img_slice_name},{msk_slice_name}\n')
