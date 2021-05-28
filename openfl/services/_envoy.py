import numpy as np
from click import echo
from longliving.repository_keeper import shard_description, keeper


#  ShardDescriptor, RepositoryKeeper

class RealShardDescriptor(shard_description.ShardDescriptor):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_path = data_path
        self.dataset_length = 100

    def __len__(self):
        return self.dataset_length

    def get_item(self, index: int):
        return None

    @property
    def sample_shape(self) -> int:
        return 1

    @property
    def target_shape(self) -> int:
        return 1

    @property
    def dataset_description(self) -> str:
        return self.data_path


def main():
    data_path = '/some/path'
    shard_descriptor = RealShardDescriptor(data_path)

    shard_name = 'third'
    director_uri = 'localhost:50051'
    repo_keeper = keeper.CollaboratorService(shard_name, director_uri)

    repo_keeper.start(shard_descriptor)


if __name__ == '__main__':
    main()
