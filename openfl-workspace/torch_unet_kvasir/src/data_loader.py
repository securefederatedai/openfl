# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import zipfile
from os import listdir
from pathlib import Path

import numpy as np
import PIL
from skimage import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tsf
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from openfl.federated import PyTorchDataLoader
from openfl.utilities import validate_file_hash


def read_data(image_path, mask_path):
    """Read image and mask from disk.

    Args:
        image_path: Path to image
        mask_path:  Path to mask

    Returns:
        Numpy image and mask

    """
    img = io.imread(image_path)
    assert (img.shape[2] == 3)
    mask = io.imread(mask_path)
    return (img, mask[:, :, 0].astype(np.uint8))


class KvasirDataset(Dataset):
    """Kvasir dataset. Splits data by shards for each collaborator."""

    def __init__(self, is_validation, shard_num, collaborator_count, **kwargs):
        """Initialize dataset.

        Args:
            is_validation: Validation dataset or not
            shard_num: Number of collaborator for which the data is splited
            collaborator_count: Total number of collaborators

        """
        self.images_path = './data/segmented-images/images/'
        self.masks_path = './data/segmented-images/masks/'
        self.images_names = [
            img_name
            for img_name in sorted(listdir(self.images_path))
            if len(img_name) > 3 and img_name[-3:] == 'jpg'
        ]

        self.images_names = self.images_names[shard_num:: collaborator_count]
        self.is_validation = is_validation
        assert (len(self.images_names) > 8)
        validation_size = len(self.images_names) // 8

        if is_validation:
            self.images_names = self.images_names[-validation_size:]
        else:
            self.images_names = self.images_names[: -validation_size]

        self.img_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332)),
            tsf.ToTensor(),
            tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332), interpolation=PIL.Image.NEAREST),
            tsf.ToTensor()])

    def __getitem__(self, index):
        """Get items by slice index."""
        name = self.images_names[index]
        img, mask = read_data(self.images_path + name, self.masks_path + name)
        img = self.img_trans(img)
        mask = self.mask_trans(mask)
        return img, mask

    def __len__(self):
        """Length of spltted data."""
        return len(self.images_names)


def load_kvasir_dataset():
    """Load and unzip kvasir dataset."""
    zip_sha384 = ('66cd659d0e8afd8c83408174'
                  '1ade2b75dada8d4648b816f2533c8748b1658efa3d49e205415d4116faade2c5810e241e')
    data_url = ('https://datasets.simula.no/downloads/'
                'hyper-kvasir/hyper-kvasir-segmented-images.zip')
    filename = 'kvasir.zip'
    data_folder_path = Path.cwd().absolute() / 'data'
    kvasir_archive_path = data_folder_path / filename
    if not kvasir_archive_path.is_file():
        download_url(data_url, data_folder_path, filename=filename)
        validate_file_hash(kvasir_archive_path, zip_sha384)
        with zipfile.ZipFile(kvasir_archive_path, 'r') as zip_ref:
            for member in tqdm(iterable=zip_ref.infolist(), desc='Unzipping dataset'):
                zip_ref.extract(member, './data')


class PyTorchKvasirDataLoader(PyTorchDataLoader):
    """PyTorch data loader for Kvasir dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        load_kvasir_dataset()
        self.valid_dataset = KvasirDataset(True, shard_num=int(data_path), **kwargs)
        self.train_dataset = KvasirDataset(False, shard_num=int(data_path), **kwargs)
        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()
        self.batch_size = batch_size

    def get_valid_loader(self, num_batches=None):
        """Return validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_data_size(self):
        """Return size of train dataset."""
        return len(self.train_dataset)

    def get_valid_data_size(self):
        """Return size of validation dataset."""
        return len(self.valid_dataset)

    def get_feature_shape(self):
        """Return data shape."""
        return self.valid_dataset[0][0].shape
