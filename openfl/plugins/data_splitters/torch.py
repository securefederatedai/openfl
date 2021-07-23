"""openfl.plugins.data_splitters.torch module."""
from abc import abstractmethod
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset

from openfl.plugins.data_splitters.data_splitter import DataSplitter
from openfl.plugins.data_splitters.numpy import DirichletNumPyDataSplitter
from openfl.plugins.data_splitters.numpy import LogNormalNumPyDataSplitter


class PyTorchDatasetSplitter(DataSplitter):
    """Base class for splitting PyTorch Datasets."""

    @abstractmethod
    def split(self, data: Dataset, num_collaborators: int) -> List[Dataset]:
        """Split the data."""
        raise NotImplementedError


class EqualPyTorchDatasetSplitter(PyTorchDatasetSplitter):
    """PyTorch Dataset version of equal dataset split."""

    def split(self, data, num_collaborators):
        """Split the data."""
        return [Subset(data,
                       np.arange(start=shard_num, stop=len(data), step=num_collaborators))
                for shard_num in range(num_collaborators)]


class LogNormalPyTorchDatasetSplitter(PyTorchDatasetSplitter):
    """Pytorch Dataset-based implementation of lognormal data split."""

    def __init__(self,
                 mu,
                 sigma,
                 num_classes,
                 classes_per_col=2,
                 min_samples_per_class=5):
        """Initialize.

        Args:
            mu(float): Distribution hyperparameter.
            sigma(float): Distribution hyperparameter.
            classes_per_col(int): Number of classes assigned to each collaborator.
            min_samples_per_class(int): Minimum number of collaborator samples of each class.
        """
        self.num_classes = num_classes
        self.numpy_splitter = LogNormalNumPyDataSplitter(mu,
                                                         sigma,
                                                         num_classes,
                                                         classes_per_col,
                                                         min_samples_per_class)

    def split(self, data, num_collaborators):
        """Split the data."""
        labels = [label for _, label in data]
        labels = np.array([y.numpy() if isinstance(y, torch.Tensor) else y for y in labels])
        flat_labels = labels.argmax(axis=1) if len(labels.shape) > 1 else labels
        idx = self.numpy_splitter.get_indices(flat_labels, num_collaborators)
        datasets = [Subset(data, col_idx) for col_idx in idx]
        return datasets


class RandomPyTorchDatasetSplitter(PyTorchDatasetSplitter):
    """PyTorch Dataset version of random dataset split."""

    def split(self, data, num_collaborators):
        """Split the data."""
        idx = np.sort(np.random.choice(len(data), num_collaborators - 1, replace=False))
        subsets = []
        for i in range(len(idx)):
            idx_range = np.arange(start=idx[i - 1] if i > 0 else 0, stop=idx[i])
            subsets.append(Subset(data, idx_range))
        last_range = np.arange(start=idx[-1], stop=len(data))
        subsets.append(Subset(data, last_range))
        return subsets


class DirichletPyTorchDatasetSplitter(PyTorchDatasetSplitter):
    """PyTorch dataset version of dirichlet split."""

    def __init__(self, alpha=0.5, min_samples_per_col=10):
        """Initialize."""
        self.numpy_splitter = DirichletNumPyDataSplitter(alpha, min_samples_per_col)

    def split(self, data, num_collaborators):
        """Split the data."""
        labels = np.array([label for _, label in data])
        idx_batch = self.numpy_splitter.split_dirichlet(labels, num_collaborators)
        return [Subset(data, idx) for idx in idx_batch]
