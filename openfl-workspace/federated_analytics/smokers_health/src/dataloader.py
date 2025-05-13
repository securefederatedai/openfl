from openfl.federated.data.loader import DataLoader
import pandas as pd
import os
import subprocess


class SmokersHealthDataLoader(DataLoader):
    """Data Loader for Smokers Health Dataset."""

    def __init__(self, batch_size, data_path, **kwargs):
        super().__init__(**kwargs)

        # Download and prepare data
        self._download_raw_data()
        self.data_shard = self.load_data_shard(
            shard_num=int(data_path), **kwargs
        )

    def _download_raw_data(self):
        """Download the dataset using curl."""
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
        """Load a specific shard of the dataset."""
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