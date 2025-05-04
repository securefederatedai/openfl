# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""


from src.runner_fa import FederatedAnalyticsTaskRunner
import numpy as np

class IrisHistogram(FederatedAnalyticsTaskRunner):
    """
    Taskrunner class used to perform federated analytics on the Iris dataset by generating histograms for specified columns.
    Methods
    -------
    __init__(**kwargs)
        Initializes the IrisHistogram instance with the provided keyword arguments.
    analytics(columns, **kwargs)
        Performs analytics on the specified columns and returns a dictionary of histograms.
    compute_hist(df, col_name)
        Computes the histogram for a specified column in the dataframe.
    """


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analytics_task(self, columns, **kwargs):
        """
        Perform analytics on the specified columns and compute histograms.
        Args:
            columns (list): List of column names to analyze.
            **kwargs: Additional keyword arguments.
        Returns:
            dict: A dictionary where keys are column names and values are histograms.
        """
        # query data
        data = self.data_loader.query(columns)
        histograms = {}
        for column in columns:
            hist, bins = self.compute_hist(data, column)
            histograms[column + " histogram"] = hist
            histograms[column + " bins"] = bins
        return histograms

    def compute_hist(self, data, col_name):
        """
        Compute the histogram of a specified column in a DataFrame.
        Args:
            data (pandas.DataFrame): The DataFrame containing the data.
            col_name (str): The name of the column for which to compute the histogram.
        Returns:
            tuple: A tuple containing the histogram and bin edges as numpy arrays.
        """
        hist, bins = np.histogram(data[col_name], bins=np.linspace(2.0, 10.0, 10))
        return hist, bins
