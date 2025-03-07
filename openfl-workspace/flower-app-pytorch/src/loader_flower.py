# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FlowerDataLoader module."""

from openfl.federated.data.loader import DataLoader


class FlowerDataLoader(DataLoader):
    """Flower Dataloader

    This class extends the OpenFL DataLoader to provide functionality for
    loading and partitioning data for a Flower workload.

    Attributes:
        data_shard (int): The shard number of the dataset.
        num_partitions (int): The number of partitions to divide the dataset into.
    """

    def __init__(self, data_path, **kwargs):
        """
        Initialize the FlowerDataLoader.

        Args:
            data_path (str or int): The directory of the dataset.
            collaborator_count (int): The number of partitions to divide the dataset into.
            **kwargs: Additional keyword arguments to pass to the parent DataLoader class.

        Raises:
            ValueError: If collaborator_count is not provided or if data_path is not a number.
        """
        super().__init__(**kwargs)
        self.data_path = data_path
    
    def get_node_configs(self):
        """
        Get the configuration for each node.

        This method returns the number of partitions and the data shard,
        which can be used by each node to access the dataset.

        Returns:
            tuple: A tuple containing the number of partitions and the data shard.
        """
        return self.data_path
    
    def get_feature_shape(self):
        """
        Override the parent method to return None.
        Flower's own infrastructure will handle the feature shape.
        """
        return None