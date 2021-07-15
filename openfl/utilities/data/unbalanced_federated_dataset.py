"""UnbalancedFederatedDataset module."""

import numpy as np
from tqdm import trange

from openfl.federated import FederatedDataSet


def get_label_count(labels, label):
    """Count samples with label `label` in `labels` array."""
    return len(np.nonzero(labels == label)[0])


class LogNormallyDistributedFederatedDataset(FederatedDataSet):
    """Class for unbalanced (LogNormal) dataset split."""

    def split(self,
              num_collaborators,
              mu=0,
              sigma=2.,
              train_classes_per_col=2,
              min_train_samples_per_class=5,
              valid_classes_per_col=2,
              min_valid_samples_per_class=5):
        """Split the data."""
        y_train, y_valid = (np.array(y) for y in [self.y_train, self.y_valid])
        X_train = np.array([np.array(X_i) for X_i in self.X_train])
        X_valid = np.array([np.array(X_i) for X_i in self.X_valid])

        train_idx = self.split_lognormal(y_train, mu, sigma, num_collaborators,
                                         train_classes_per_col, min_train_samples_per_class)
        X_train = np.array([X_train[idx] for idx in train_idx])
        y_train = np.array([y_train[idx] for idx in train_idx])

        valid_idx = self.split_lognormal(y_valid, mu, sigma, num_collaborators,
                                         valid_classes_per_col, min_valid_samples_per_class)
        X_valid = np.array([X_valid[idx] for idx in valid_idx])
        y_valid = np.array([y_valid[idx] for idx in valid_idx])
        return [
            FederatedDataSet(
                X_train[i],
                y_train[i],
                X_valid[i],
                y_valid[i],
                batch_size=self.batch_size,
                num_classes=self.num_classes
            ) for i in range(num_collaborators)
        ]

    def split_lognormal(self,
                        labels,
                        mu,
                        sigma,
                        num_collaborators,
                        classes_per_col,
                        min_samples_per_class):
        """Split labels into unequal parts by lognormal law.

        Args:
            labels(np.ndarray): Array of class labels.
            mu(float): Distribution hyperparameter.
            sigma(float): Distribution hyperparameter.
            num_collaborators(int): Number of data slices.
            classes_per_col(int): Number of classes assigned to each collaborator.
            min_samples_per_class(int): Minimum number of collaborator samples of each class.

        Returns:
            np.ndarray: Array of arrays of data indices assigned per collaborator.
        """
        labels = np.array(labels)
        idx = [[] for _ in range(num_collaborators)]
        samples_per_col = classes_per_col * min_samples_per_class
        for col in range(num_collaborators):
            for j in range(classes_per_col):
                label = (col + j) % self.num_classes
                label_idx = np.nonzero(labels == label)[0]
                slice_start = col // self.num_classes * samples_per_col + min_samples_per_class * j
                slice_end = slice_start + min_samples_per_class
                print(f'Assigning {slice_start}:{slice_end} of {label} class to {col} col...')
                idx[col] += list(label_idx[slice_start:slice_end])
        assert all([len(i) == samples_per_col for i in idx]), \
            f'All collaborators should have {classes_per_col * min_samples_per_class} elements'

        props_shape = (self.num_classes, num_collaborators // 10, classes_per_col)
        props = np.random.lognormal(mu, sigma, props_shape)
        num_samples_per_class = [[[get_label_count(labels, label) - min_samples_per_class]]
                                 for label in range(self.num_classes)]
        num_samples_per_class = np.array(num_samples_per_class)
        props = num_samples_per_class * props / np.sum(props, (1, 2), keepdims=True)
        for user in trange(num_collaborators):
            for j in range(classes_per_col):
                label = (user + j) % self.num_classes
                num_samples = int(props[label, user // 10, j])

                print(f'Trying to append {num_samples} of {label} class to {user} col...')
                slice_start = np.count_nonzero(labels[np.hstack(idx)] == label)
                slice_end = slice_start + num_samples
                if slice_end < get_label_count(labels, label):
                    label_subset = np.nonzero(labels == (user + j) % self.num_classes)[0]
                    idx_to_append = label_subset[slice_start:slice_end]
                    print(f'Appending {idx_to_append} of {label} class to {user} col...')
                    idx[user] = np.append(idx[user], idx_to_append)
        return idx


class DataLoaderLogNormallyDistributedFederatedDataset(LogNormallyDistributedFederatedDataset):
    """Pytorch Dataset-based implementation of lognormal data split."""

    def split(self,
              num_collaborators,
              mu=0,
              sigma=2.,
              train_classes_per_col=2,
              min_train_samples_per_class=5,
              valid_classes_per_col=2,
              min_valid_samples_per_class=5):
        """Split the data."""
        self.X_train, self.y_train = list(zip(*self.training_set))
        self.X_valid, self.y_valid = list(zip(*self.valid_set))
        return super().split(num_collaborators,
                             mu,
                             sigma,
                             train_classes_per_col,
                             min_train_samples_per_class,
                             valid_classes_per_col,
                             min_valid_samples_per_class)
