# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""You may copy this file as the starting point of your own model."""

import logging
import os

import numpy as np

from src.nii_reader import nii_reader

logger = logging.getLogger(__name__)


def train_val_split(features, labels, percent_train, shuffle):
    """Train/validation splot of the BraTS dataset.

    Splits incoming feature and labels into training and validation. The value
    of shuffle determines whether shuffling occurs before the split is performed.

    Args:
        features: The input images
        labels: The ground truth labels
        percent_train (float): The percentage of the dataset that is training.
        shuffle (bool): True = shuffle the dataset before the split

    Returns:
        train_features: The input images for the training dataset
        train_labels: The ground truth labels for the training dataset
        val_features: The input images for the validation dataset
        val_labels: The ground truth labels for the validation dataset
    """

    def split(lst, idx):
        """Split a Python list into 2 lists.

        Args:
            lst: The Python list to split
            idx: The index where to split the list into 2 parts

        Returns:
            Two lists

        """
        if idx < 0 or idx > len(lst):
            raise ValueError('split was out of expected range.')
        return lst[:idx], lst[idx:]

    nb_features = len(features)
    nb_labels = len(labels)
    if nb_features != nb_labels:
        raise RuntimeError('Number of features and labels do not match.')
    if shuffle:
        new_order = np.random.permutation(np.arange(nb_features))
        features = features[new_order]
        labels = labels[new_order]
    split_idx = int(percent_train * nb_features)
    train_features, val_features = split(lst=features, idx=split_idx)
    train_labels, val_labels = split(lst=labels, idx=split_idx)
    return train_features, train_labels, val_features, val_labels


def load_from_nifti(parent_dir,
                    percent_train,
                    shuffle,
                    channels_last=True,
                    task='whole_tumor',
                    **kwargs):
    """Load the BraTS dataset from the NiFTI file format.

    Loads data from the parent directory (NIfTI files for whole brains are
    assumed to be contained in subdirectories of the parent directory).
    Performs a split of the data into training and validation, and the value
    of shuffle determined whether shuffling is performed before this split
    occurs - both split and shuffle are done in a way to
    keep whole brains intact. The kwargs are passed to nii_reader.

    Args:
        parent_dir: The parent directory for the BraTS data
        percent_train (float): The percentage of the data to make the training dataset
        shuffle (bool): True means shuffle the dataset order before the split
        channels_last (bool): Input tensor uses channels as last dimension (Default is True)
        task: Prediction task (Default is 'whole_tumor' prediction)
        **kwargs: Variable arguments to pass to the function

    Returns:
        train_features: The input images for the training dataset
        train_labels: The ground truth labels for the training dataset
        val_features: The input images for the validation dataset
        val_labels: The ground truth labels for the validation dataset

    """
    path = os.path.join(parent_dir)
    subdirs = os.listdir(path)
    subdirs.sort()
    if not subdirs:
        raise SystemError(f'''{parent_dir} does not contain subdirectories.
Please make sure you have BraTS dataset downloaded
and located in data directory for this collaborator.
        ''')
    subdir_paths = [os.path.join(path, subdir) for subdir in subdirs]

    imgs_all = []
    msks_all = []
    for brain_path in subdir_paths:
        these_imgs, these_msks = nii_reader(
            brain_path=brain_path,
            task=task,
            channels_last=channels_last,
            **kwargs
        )
        # the needed files where not present if a tuple of None is returned
        if these_imgs is None:
            logger.debug(f'Brain subdirectory: {brain_path} did not contain the needed files.')
        else:
            imgs_all.append(these_imgs)
            msks_all.append(these_msks)

    # converting to arrays to allow for numpy indexing used during split
    imgs_all = np.array(imgs_all)
    msks_all = np.array(msks_all)

    # note here that each is a list of 155 slices per brain, and so the
    # split keeps brains intact
    imgs_all_train, msks_all_train, imgs_all_val, msks_all_val = train_val_split(
        features=imgs_all,
        labels=msks_all,
        percent_train=percent_train,
        shuffle=shuffle
    )
    # now concatenate the lists
    imgs_train = np.concatenate(imgs_all_train, axis=0)
    msks_train = np.concatenate(msks_all_train, axis=0)
    imgs_val = np.concatenate(imgs_all_val, axis=0)
    msks_val = np.concatenate(msks_all_val, axis=0)

    return imgs_train, msks_train, imgs_val, msks_val
