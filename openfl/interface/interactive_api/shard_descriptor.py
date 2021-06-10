import numpy as np


class ShardDescriptor:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        # -> Tuple(np.ndarray, np.ndarray)
        raise NotImplementedError

    @property
    def sample_shape(self):
        # int( sum( [str(dim) for dim in sample.shape] ) )
        raise NotImplementedError

    @property
    def target_shape(self):
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        return ''


class DummyShardDescriptor(ShardDescriptor):

    def __init__(self, sample_shape, target_shape, size) -> None:
        self._sample_shape = [int(dim) for dim in sample_shape]
        self._target_shape = [int(dim) for dim in target_shape]
        self.size = size
        self.samples = np.random.randint(0, 255, (self.size, *self.sample_shape), np.uint8)
        self.targets = np.random.randint(0, 255, (self.size, *self.target_shape), np.uint8)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # -> Tuple(np.ndarray, np.ndarray)
        return self.samples[index], self.targets[index]

    @property
    def sample_shape(self):
        # int( sum( [str(dim) for dim in sample.shape] ) )
        return self._sample_shape

    @property
    def target_shape(self):
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        return 'Dummy shard descriptor'
