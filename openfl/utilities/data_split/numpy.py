"""UnbalancedFederatedDataset module."""

from abc import abstractmethod
from typing import Dict
from typing import Tuple

import numpy as np
from tqdm import trange

from openfl.utilities.data_split.data_splitter import DataSplitter


def get_label_count(labels, label):
    """Count samples with label `label` in `labels` array."""
    return len(np.nonzero(labels == label)[0])


def one_hot(labels, classes):
    """Apply One-Hot encoding to labels."""
    return np.eye(classes)[labels]


class NumPyDataSplitter(DataSplitter):
    """Base class for splitting numpy arrays of data."""

    @abstractmethod
    def split(self, data: Tuple[np.ndarray, np.ndarray], num_collaborators: int) \
            -> Dict[str, np.ndarray]:
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
        samples, labels = data
        if self.shuffle:
            shuffled_idx = np.random.choice(
                len(labels), len(labels), replace=False
            )
            samples = samples[shuffled_idx]
            labels = labels[shuffled_idx]
        X = np.array_split(samples, num_collaborators)
        y = np.array_split(labels, num_collaborators)
        return [{'data': samples, 'labels': labels} for samples, labels in zip(X, y)]


class RandomNumPyDataSplitter(NumPyDataSplitter):
    """Splits the data randomly."""

    def __init__(self, shuffle=True):
        """Initialize.

        Args:
            shuffle(bool): Flag determining whether to shuffle the dataset before splitting.
        """
        self.shuffle = shuffle

    def split(self, data, num_collaborators):
        """Split the data."""
        samples, labels = data
        if self.shuffle:
            shuffled_idx = np.random.choice(
                len(labels), len(labels), replace=False
            )
            samples = samples[shuffled_idx]
            labels = labels[shuffled_idx]
        random_idx = np.sort(np.random.choice(
            len(labels), num_collaborators - 1, replace=False)
        )
        X = np.split(samples, random_idx)
        y = np.split(labels, random_idx)
        return [{'data': samples, 'labels': labels} for samples, labels in zip(X, y)]


class LogNormalNumPyDataSplitter(NumPyDataSplitter):
    """Unbalanced (LogNormal) dataset split."""

    def __init__(self, mu,
                 sigma,
                 num_classes,
                 classes_per_col,
                 min_samples_per_class):
        """Initialize.

        Args:
            mu(float): Distribution hyperparameter.
            sigma(float): Distribution hyperparameter.
            classes_per_col(int): Number of classes assigned to each collaborator.
            min_samples_per_class(int): Minimum number of collaborator samples of each class.
        """
        self.mu = mu
        self.sigma = sigma
        self.num_classes = num_classes
        self.classes_per_col = classes_per_col
        self.min_samples_per_class = min_samples_per_class

    def split(self, data, num_collaborators):
        """Split the data."""
        samples, labels = data
        idx = self.get_indices(labels, num_collaborators)
        return [{'data': samples[i], 'labels':labels[i]} for i in idx]

    def get_indices(self,
                    labels,
                    num_collaborators):
        """Split labels into unequal parts by lognormal law.

        Args:
            labels(np.ndarray): Array of class labels.
            num_collaborators(int): Number of data slices.
        Returns:
            np.ndarray: Array of arrays of data indices assigned per collaborator.
        """
        idx = [[] for _ in range(num_collaborators)]
        samples_per_col = self.classes_per_col * self.min_samples_per_class
        for col in range(num_collaborators):
            for c in range(self.classes_per_col):
                label = (col + c) % self.num_classes
                label_idx = np.nonzero(labels == label)[0]
                slice_start = col // self.num_classes * samples_per_col
                slice_start += self.min_samples_per_class * c
                slice_end = slice_start + self.min_samples_per_class
                print(f'Assigning {slice_start}:{slice_end} of {label} class to {col} col...')
                idx[col] += list(label_idx[slice_start:slice_end])
        assert all([len(i) == samples_per_col for i in idx]), \
            f'All collaborators should have {samples_per_col} elements' \
            + f'but distribution is {[len(i) for i in idx]}'

        props_shape = (self.num_classes, num_collaborators // 10, self.classes_per_col)
        props = np.random.lognormal(self.mu, self.sigma, props_shape)
        num_samples_per_class = [[[get_label_count(labels, label) - self.min_samples_per_class]]
                                 for label in range(self.num_classes)]
        num_samples_per_class = np.array(num_samples_per_class)
        props = num_samples_per_class * props / np.sum(props, (1, 2), keepdims=True)
        for col in trange(num_collaborators):
            for j in range(self.classes_per_col):
                label = (col + j) % self.num_classes
                num_samples = int(props[label, col // 10, j])

                print(f'Trying to append {num_samples} of {label} class to {col} col...')
                slice_start = np.count_nonzero(labels[np.hstack(idx)] == label)
                slice_end = slice_start + num_samples
                if slice_end < get_label_count(labels, label):
                    label_subset = np.nonzero(labels == (col + j) % self.num_classes)[0]
                    idx_to_append = label_subset[slice_start:slice_end]
                    print(f'Appending {idx_to_append} of {label} class to {col} col...')
                    idx[col] = np.append(idx[col], idx_to_append)
        return idx
