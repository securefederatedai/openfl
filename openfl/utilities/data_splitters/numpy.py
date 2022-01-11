"""UnbalancedFederatedDataset module."""

from abc import abstractmethod
from typing import List

import numpy as np
from tqdm import trange

from openfl.utilities.data_splitters.data_splitter import DataSplitter


def get_label_count(labels, label):
    """Count samples with label `label` in `labels` array."""
    return len(np.nonzero(labels == label)[0])


def one_hot(labels, classes):
    """Apply One-Hot encoding to labels."""
    return np.eye(classes)[labels]


class NumPyDataSplitter(DataSplitter):
    """Base class for splitting numpy arrays of data."""

    @abstractmethod
    def split(self, data: np.ndarray, num_collaborators: int) -> List[List[int]]:
        """Split the data."""
        raise NotImplementedError


class EqualNumPyDataSplitter(NumPyDataSplitter):
    """Splits the data evenly."""

    def __init__(self, shuffle=True):
        """Initialize.

        Args:
            shuffle(bool): Flag determining whether to shuffle the dataset before splitting.
        """
        self.shuffle = shuffle

    def split(self, data, num_collaborators):
        """Split the data."""
        idx = range(len(data))
        if self.shuffle:
            idx = np.random.permutation(idx)
        slices = np.array_split(idx, num_collaborators)
        return slices


class RandomNumPyDataSplitter(NumPyDataSplitter):
    """Splits the data randomly."""

    def __init__(self, shuffle=True, seed=0):
        """Initialize.

        Args:
            shuffle(bool): Flag determining whether to shuffle the dataset before splitting.
            seed(int): Random numbers generator seed.
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
    """Unbalanced (LogNormal) dataset split."""

    def __init__(self, mu,
                 sigma,
                 num_classes,
                 classes_per_col,
                 min_samples_per_class,
                 seed=0):
        """Initialize.

        Args:
            mu(float): Distribution hyperparameter.
            sigma(float): Distribution hyperparameter.
            classes_per_col(int): Number of classes assigned to each collaborator.
            min_samples_per_class(int): Minimum number of collaborator samples of each class.
            seed(int): Random numbers generator seed.
        """
        self.mu = mu
        self.sigma = sigma
        self.num_classes = num_classes
        self.classes_per_col = classes_per_col
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data."""
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
                print(f'Assigning {slice_start}:{slice_end} of {label} class to {col} col...')
                idx[col] += list(label_idx[slice_start:slice_end])
        assert all([len(i) == samples_per_col for i in idx]), f'''
All collaborators should have {samples_per_col} elements
but distribution is {[len(i) for i in idx]}'''

        props_shape = (self.num_classes, num_collaborators // 10, self.classes_per_col)
        props = np.random.lognormal(self.mu, self.sigma, props_shape)
        num_samples_per_class = [[[get_label_count(data, label) - self.min_samples_per_class]]
                                 for label in range(self.num_classes)]
        num_samples_per_class = np.array(num_samples_per_class)
        props = num_samples_per_class * props / np.sum(props, (1, 2), keepdims=True)
        for col in trange(num_collaborators):
            for j in range(self.classes_per_col):
                label = (col + j) % self.num_classes
                num_samples = int(props[label, col // 10, j])

                print(f'Trying to append {num_samples} of {label} class to {col} col...')
                slice_start = np.count_nonzero(data[np.hstack(idx)] == label)
                slice_end = slice_start + num_samples
                if slice_end < get_label_count(data, label):
                    label_subset = np.nonzero(data == (col + j) % self.num_classes)[0]
                    idx_to_append = label_subset[slice_start:slice_end]
                    print(f'Appending {idx_to_append} of {label} class to {col} col...')
                    idx[col] = np.append(idx[col], idx_to_append)
        return idx


class DirichletNumPyDataSplitter(NumPyDataSplitter):
    """Numpy splitter according to dirichlet distribution."""

    def __init__(self, alpha=0.5, min_samples_per_col=10, seed=0):
        """Initialize."""
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
                proportions = [p * (len(idx_j) < n / num_collaborators)
                               for p, idx_j in zip(proportions, idx_batch)]
                proportions = np.array(proportions)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_splitted = np.split(idx_k, proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_splitted)]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        return idx_batch
