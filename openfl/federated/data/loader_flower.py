# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FlowerDataLoader module."""

import os

from openfl.federated.data.loader import DataLoader


class FlowerDataLoader(DataLoader):
    """Flower Dataloader

    This class extends the OpenFL DataLoader to provide functionality for
    loading and partitioning data for a Flower workload.

    """

    def __init__(self, data_path, **kwargs):
        """
        Initialize the FlowerDataLoader.

        Args:
            data_path (str or int): The directory of the dataset.
            **kwargs: Additional keyword arguments to pass to the parent DataLoader class.

        Raises:
            FileNotFoundError: If the specified data path does not exist.
        """
        super().__init__(**kwargs)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

        self.data_path = data_path

    def get_node_configs(self):
        """
        Get the configuration for each node.

        This method returns the number of partitions and the data shard,
        which can be used by each node to access the dataset.

        Returns:
            str: data path
        """
        return self.data_path

    def get_feature_shape(self):
        """
        Override the parent method to return None.
        Flower's own infrastructure will handle the feature shape.
        """
        return None
