import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import Sequence

from openfl.interface.interactive_api.experiment import DataInterface


class DataGenerator(Sequence):

    def __init__(self, shard_descriptor, batch_size):
        self.shard_descriptor = shard_descriptor
        self.batch_size = batch_size
        self.indices = np.arange(len(shard_descriptor))
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.shard_descriptor[batch]
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class FedDataset(DataInterface):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor

    def __getitem__(self, index):
        return self.shard_descriptor[index]

    def __len__(self):
        return len(self.shard_descriptor)

    def get_train_loader(self):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        if self.kwargs['train_bs']:
            batch_size = self.kwargs['train_bs']
        else:
            batch_size = 32
        self.shard_descriptor.set_dataset_type(mode='train')
        return DataGenerator(self.shard_descriptor, batch_size=batch_size)

    def get_valid_loader(self):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        if self.kwargs['valid_bs']:
            batch_size = self.kwargs['valid_bs']
        else:
            batch_size = 32
        
        self.shard_descriptor.set_dataset_type(mode='val')
        return DataGenerator(self.shard_descriptor, batch_size=batch_size)

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        
        return self.shard_descriptor.get_train_size()

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return self.shard_descriptor.get_test_size()
