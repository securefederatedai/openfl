# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""UnbalancedFederatedDataset module."""

from abc import abstractmethod
from typing import List

import numpy as np
from tqdm import trange

from openfl.utilities.data_splitters.data_splitter import DataSplitter


def get_label_count(labels, label):
    """Count the number of samples with a specific label in a labels array.

    Args:
        labels (np.ndarray): Array of labels.
        label (int or str): The label to count.

    Returns:
        int: The count of the label in the labels array.
    """
    return len(np.nonzero(labels == label)[0])


def one_hot(labels, classes):
    """Apply One-Hot encoding to labels.

    Args:
        labels (np.ndarray): Array of labels.
        classes (int): The total number of classes.

    Returns:
        np.ndarray: The one-hot encoded labels.
    """
    return np.eye(classes)[labels]


class NumPyDataSplitter(DataSplitter):
    """Base class for splitting numpy arrays of data.

    This class should be subclassed when creating specific data splitter
    classes.
    """

    @abstractmethod
    def split(self, data: np.ndarray, num_collaborators: int) -> List[List[int]]:
        """Split the data."""
        raise NotImplementedError


class EqualNumPyDataSplitter(NumPyDataSplitter):
    """Class for splitting numpy arrays of data evenly.

    Args:
        shuffle (bool, optional): Flag determining whether to shuffle the
            dataset before splitting. Defaults to True.
        seed (int, optional): Random numbers generator seed. Defaults to 0.
    """

    def __init__(self, shuffle=True, seed=0):
        """Initialize.

        Args:
            shuffle (bool): Flag determining whether to shuffle the dataset
                before splitting. Defaults to True.
            seed (int): Random numbers generator seed. Defaults to 0.
                For different splits on envoys, try setting different values
                for this parameter on each shard descriptor.
        """
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data."""
        np.random.seed(self.seed)
        idx = range(len(data))
        if self.shuffle:
            idx = np.random.permutation(idx)
        slices = np.array_split(idx, num_collaborators)
        return slices


class RandomNumPyDataSplitter(NumPyDataSplitter):
    """Class for splitting numpy arrays of data randomly.

    Args:
        shuffle (bool, optional): Flag determining whether to shuffle the
            dataset before splitting. Defaults to True.
        seed (int, optional): Random numbers generator seed. Defaults to 0.
    """

    def __init__(self, shuffle=True, seed=0):
        """Initialize.

        Args:
            shuffle (bool): Flag determining whether to shuffle the dataset
                before splitting. Defaults to True.
            seed (int): Random numbers generator seed. Defaults to 0.
                For different splits on envoys, try setting different values
                for this parameter on each shard descriptor.
        """
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data."""
        np.random.seed(self.seed)
        idx = range(len(data))
        if self.shuffle:
            idx = np.random.permutation(idx)
        random_idx = np.sort(np.random.choice(len(data), num_collaborators - 1, replace=False))

        return np.split(idx, random_idx)


class LogNormalNumPyDataSplitter(NumPyDataSplitter):
    """Class for splitting numpy arrays of data according to a LogNormal
    distribution.

    Unbalanced (LogNormal) dataset split.
    This split assumes only several classes are assigned to each collaborator.
    Firstly, it assigns classes_per_col * min_samples_per_class items of
    dataset to each collaborator so all of collaborators will have some data
    after the split.
    Then, it generates positive integer numbers by log-normal (power) law.
    These numbers correspond to numbers of dataset items picked each time from
    dataset and assigned to a collaborator.
    Generation is repeated for each class assigned to a collaborator.
    This is a parametrized version of non-i.i.d. data split in FedProx
    algorithm.
    Origin source: https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py#L30

    Args:
        mu (float): Distribution hyperparameter.
        sigma (float): Distribution hyperparameter.
        num_classes (int): Number of classes.
        classes_per_col (int): Number of classes assigned to each collaborator.
        min_samples_per_class (int): Minimum number of collaborator samples of
            each class.
        seed (int, optional): Random numbers generator seed. Defaults to 0.

    .. note::
        This split always drops out some part of the dataset!
        Non-deterministic behavior selects only random subpart of class items.
    """

    def __init__(
        self,
        mu,
        sigma,
        num_classes,
        classes_per_col,
        min_samples_per_class,
        seed=0,
    ):
        """Initialize the generator.

        Args:
            mu (float): Distribution hyperparameter.
            sigma (float): Distribution hyperparameter.
            classes_per_col (int): Number of classes assigned to each
                collaborator.
            min_samples_per_class (int): Minimum number of collaborator
                samples of each class.
            seed (int): Random numbers generator seed. Defaults to 0.
                For different splits on envoys, try setting different values
                for this parameter on each shard descriptor.
        """
        self.mu = mu
        self.sigma = sigma
        self.num_classes = num_classes
        self.classes_per_col = classes_per_col
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data.

        Args:
            data (np.ndarray): numpy-like label array.
            num_collaborators (int): number of collaborators to split data
                across.
                Should be divisible by number of classes in ``data``.
        """
        np.random.seed(self.seed)
        idx = [[] for _ in range(num_collaborators)]
        samples_per_col = self.classes_per_col * self.min_samples_per_class
        for col in range(num_collaborators):
            for c in range(self.classes_per_col):
                label = (col + c) % self.num_classes
                label_idx = np.nonzero(data == label)[0]
                slice_start = col // self.num_classes * samples_per_col
                slice_start += self.min_samples_per_class * c
                slice_end = slice_start + self.min_samples_per_class
                print(f"Assigning {slice_start}:{slice_end} of class {label} to {col} col...")
                idx[col] += list(label_idx[slice_start:slice_end])
        if any(len(i) != samples_per_col for i in idx):
            raise SystemError(
                f"""All collaborators should have {samples_per_col} elements
but distribution is {[len(i) for i in idx]}"""
            )

        props_shape = (
            self.num_classes,
            num_collaborators // self.num_classes,
            self.classes_per_col,
        )
        props = np.random.lognormal(self.mu, self.sigma, props_shape)
        num_samples_per_class = [
            [[get_label_count(data, label) - self.min_samples_per_class]]
            for label in range(self.num_classes)
        ]
        num_samples_per_class = np.array(num_samples_per_class)
        props = num_samples_per_class * props / np.sum(props, (1, 2), keepdims=True)
        for col in trange(num_collaborators):
            for j in range(self.classes_per_col):
                label = (col + j) % self.num_classes
                num_samples = int(props[label, col // self.num_classes, j])

                print(f"Trying to append {num_samples} samples of {label} class to {col} col...")
                slice_start = np.count_nonzero(data[np.hstack(idx)] == label)
                slice_end = slice_start + num_samples
                label_count = get_label_count(data, label)
                if slice_end < label_count:
                    label_subset = np.nonzero(data == (col + j) % self.num_classes)[0]
                    idx_to_append = label_subset[slice_start:slice_end]
                    idx[col] = np.append(idx[col], idx_to_append)
                else:
                    print(
                        f"Index {slice_end} is out of bounds "
                        f"of array of length {label_count}. Skipping..."
                    )
        print(f"Split result: {[len(i) for i in idx]}.")
        return idx


class DirichletNumPyDataSplitter(NumPyDataSplitter):
    """Class for splitting numpy arrays of data according to a Dirichlet
    distribution.

    Generates the random sample of integer numbers from dirichlet distribution
    until minimum subset length exceeds the specified threshold.
    This behavior is a parametrized version of non-i.i.d. split in FedMA
    algorithm.
    Origin source: https://github.com/IBM/FedMA/blob/master/utils.py#L96

    Args:
        alpha (float, optional): Dirichlet distribution parameter. Defaults
            to 0.5.
        min_samples_per_col (int, optional): Minimal amount of samples per
            collaborator. Defaults to 10.
        seed (int, optional): Random numbers generator seed. Defaults to 0.
    """

    def __init__(self, alpha=0.5, min_samples_per_col=10, seed=0):
        """Initialize.

        Args:
            alpha (float): Dirichlet distribution parameter. Defaults to 0.5.
            min_samples_per_col (int): Minimal amount of samples per
                collaborator. Defaults to 10.
            seed (int): Random numbers generator seed. Defaults to 0.
                For different splits on envoys, try setting different values
                for this parameter on each shard descriptor.
        """
        self.alpha = alpha
        self.min_samples_per_col = min_samples_per_col
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data."""
        np.random.seed(self.seed)
        classes = len(np.unique(data))
        min_size = 0

        n = len(data)
        while min_size < self.min_samples_per_col:
            idx_batch = [[] for _ in range(num_collaborators)]
            for k in range(classes):
                idx_k = np.where(data == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, num_collaborators))
                proportions = [
                    p * (len(idx_j) < n / num_collaborators)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
                proportions = np.array(proportions)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_splitted = np.split(idx_k, proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_splitted)]
                min_size = min(len(idx_j) for idx_j in idx_batch)
        return idx_batch
