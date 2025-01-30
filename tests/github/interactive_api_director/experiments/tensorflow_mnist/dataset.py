import tensorflow as tf
import numpy as np

from openfl.interface.interactive_api.experiment import DataInterface


class FedDataset(DataInterface):
    def __init__(self, train_bs, valid_bs, **kwargs):
        super().__init__(**kwargs)
        self.train_bs = train_bs
        self.valid_bs = valid_bs

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        validation_size = len(self.shard_descriptor) // 10
        self.train_indices = np.arange(len(self.shard_descriptor) - validation_size)
        self.val_indices = np.arange(len(self.shard_descriptor) - validation_size,
                                     len(self.shard_descriptor))

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        samples, targets = [], []
        for i in self.train_indices:
            sample, target = self.shard_descriptor[i]
            samples.append(sample)
            targets.append(target)
        samples = np.array(samples)
        targets = np.array(targets)
        return tf.data.Dataset.from_tensor_slices((samples, targets)).batch(self.train_bs)

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        samples, targets = zip(*[self.shard_descriptor[i] for i in self.val_indices])
        samples = np.array(samples)
        targets = np.array(targets)
        return tf.data.Dataset.from_tensor_slices((samples, targets)).batch(self.valid_bs)

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_indices)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.val_indices)
