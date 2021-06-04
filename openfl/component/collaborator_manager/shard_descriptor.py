from openfl.component.collaborator_manager.shard_descriptor_base import ShardDescriptorBase


class ShardDescriptor(ShardDescriptorBase):
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
