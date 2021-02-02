# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from tqdm import tqdm
import urllib.request
import zipfile
from hashlib import sha384
from os import path
from os import listdir
from openfl.federated import PyTorchDataLoader

import PIL
import numpy as np
from skimage import io
from torchvision import transforms as tsf
from torch.utils.data import Dataset, DataLoader


def read_data(image_path, mask_path):
    """Read image and mask from disk.

    Args:
        image_path: Path to image
        mask_path:  Path to mask

    Returns:
        Numpy image and mask

    """
    img = io.imread(image_path)
    assert(img.shape[2] == 3)
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
        self.images_names = [img_name for img_name in sorted(listdir(
            self.images_path)) if len(img_name) > 3 and img_name[-3:] == 'jpg']

        self.images_names = self.images_names[shard_num:: collaborator_count]
        self.is_validation = is_validation
        assert(len(self.images_names) > 8)
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


def my_hook(t):
    """Reporthook for urlretrieve."""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def load_kvasir_dataset():
    """Load and unzip kvasir dataset."""
    ZIP_SHA384 = 'e30d18a772c6520476e55b610a4db457237f151e'\
        '19182849d54b49ae24699881c1e18e0961f77642be900450ef8b22e7'
    data_url = "https://datasets.simula.no/hyper-kvasir/hyper-kvasir-segmented-images.zip"
    filepath = './kvasir.zip'
    with tqdm(unit='B', unit_scale=True, leave=True, miniters=1,
              desc='Downloading kvasir dataset: ') as t:
        urllib.request.urlretrieve(data_url, filename=filepath,
                                   reporthook=my_hook(t), data=None)
    assert sha384(open(filepath, 'rb').read(
        path.getsize(filepath))).hexdigest() == ZIP_SHA384

    with zipfile.ZipFile(filepath, "r") as zip_ref:
        for member in tqdm(iterable=zip_ref.infolist(), desc='Unzipping dataset'):
            zip_ref.extract(member, "./data")


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
