# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.federated.data.loader import DataLoader
import pandas as pd
import os
import subprocess


class SmokersHealthDataLoader(DataLoader):
    """Data Loader for Smokers Health Dataset."""

    def __init__(self, batch_size, data_path, **kwargs):
        super().__init__(**kwargs)

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        # Load actual data if a data path is provided
        try:
            int(data_path)
        except ValueError:
            raise ValueError(
                f"Expected '{data_path}' to be representable as `int`, "
                "as it refers to the data shard number used by the collaborator."
            )

        # Download and prepare data
        self._download_raw_data()
        self.data_shard = self.load_data_shard(
            shard_num=int(data_path), **kwargs
        )

    def _download_raw_data(self):
        """
        Downloads and extracts the raw data for the smokers' health dataset.
        This method performs the following steps:
        1. Downloads the dataset from the specified Kaggle URL using the `curl` command.
        2. Saves the downloaded file as a ZIP archive in the `./data` directory.
        3. Extracts the contents of the ZIP archive into the `data` directory.
        """

        download_path = os.path.expanduser('./data/smokers_health.zip')
        subprocess.run(
            [
                'curl', '-L', '-o', download_path,
                'https://www.kaggle.com/api/v1/datasets/download/jaceprater/smokers-health-data'
            ],
            check=True
        )

        # Unzip the downloaded file into the data directory
        subprocess.run(['unzip', '-o', download_path, '-d', 'data'], check=True)

    def load_data_shard(self, shard_num, **kwargs):
        """
        Loads data from a CSV file.
        This method reads the data from a CSV file located at './data/smoking_health_data_final.csv'
        and returns it as a pandas DataFrame.
        Returns:
            pd.DataFrame: The data loaded from the CSV file.
        """
        file_path = os.path.join('data', 'smoking_health_data_final.csv')
        df = pd.read_csv(file_path)

        # Split data into shards
        shard_size = len(df) // shard_num
        start_idx = shard_size * (shard_num - 1)
        end_idx = start_idx + shard_size

        return df.iloc[start_idx:end_idx]

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
