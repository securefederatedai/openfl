# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openfl.federated import FederatedAnalyticsDataLoader
from sklearn.datasets import load_iris
import pandas as pd


class IRISInMemory(FederatedAnalyticsDataLoader):
    """Data Loader for IRIS Dataset."""

    def __init__(self, batch_size, data_path, **kwargs):
        super().__init__(**kwargs)

        # download data
        self._download_raw_data()
        # create shards
        self.data_shard = self.load_mnist_shard(
            shard_num=int(data_path), **kwargs
        )

    def _download_raw_data(self):
        iris = load_iris(as_frame=True)
        data = iris['data']
        data.to_csv('./data/client.csv', index=False)

    def _load_raw_datashards(self):
        return pd.read_csv('./data/client.csv')


    def load_mnist_shard(self, shard_num, collaborator_count, **kwargs):
        return self._load_raw_datashards().iloc[shard_num::collaborator_count]

    def query(self, columns, **kwargs):
        """
        Query the data shard for the specified columns.
        Parameters:
        columns (list): A list of column names to query from the data shard.
        **kwargs: Additional keyword arguments (currently not used).
        Returns:
        DataFrame: A DataFrame containing the data for the specified columns.
        Raises:
        ValueError: If the columns parameter is not a list.
        """
        if not isinstance(columns, list):
            raise ValueError("Columns parameter must be a list")
        return self.data_shard[columns]
