# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openfl.federated.data.loader import DataLoader
from sklearn.datasets import load_iris
import pandas as pd


class IRISInMemory(DataLoader):
    """Data Loader for IRIS Dataset."""

    def __init__(self, batch_size, data_path, **kwargs):
        super().__init__(**kwargs)

        # download data
        self._download_raw_data()
        # create shards
        self.data_shard = self.load_data_shard(
            shard_num=int(data_path), **kwargs
        )

    def _download_raw_data(self):
        """
        Downloads the raw Iris dataset and saves it as a CSV file.
        This method loads the Iris dataset using the `load_iris` function from
        the `sklearn.datasets` module. The dataset is then converted to a
        pandas DataFrame and saved as a CSV file named 'client.csv' in the
        './data/' directory.
        Returns:
            None
        """

        iris = load_iris(as_frame=True)
        data = iris['data']
        data.to_csv('./data/client.csv', index=False)

    def _load_data(self):
        """
        Loads data from a CSV file.
        This method reads the data from a CSV file located at './data/client.csv'
        and returns it as a pandas DataFrame.
        Returns:
            pd.DataFrame: The data loaded from the CSV file.
        """

        return pd.read_csv('./data/client.csv')

    def load_data_shard(self, shard_num, collaborator_count, **kwargs):
        """
        Load a specific shard of dataset for a given collaborator.
        Args:
            shard_num (int): The shard number to load.
            collaborator_count (int): The total number of collaborators.
            **kwargs: Additional keyword arguments.
        Returns:
            pandas.DataFrame: The shard of dataset corresponding to the given shard number.
        """

        return self._load_data().iloc[shard_num-1::collaborator_count]

    def query(self, columns, **kwargs):
        """
        Query the data shard for the specified columns.
        Args:
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

    def get_feature_shape(self):
        """
        This function is not required and is kept for compatibility.

        Returns:
            None
        """
        pass
