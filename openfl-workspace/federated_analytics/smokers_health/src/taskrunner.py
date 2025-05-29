# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from src.runner_fa import FederatedAnalyticsTaskRunner
import pandas as pd
import numpy as np


class SmokersHealthAnalytics(FederatedAnalyticsTaskRunner):
    """
    Taskrunner class for performing federated analytics on the Smokers Health dataset.
    Methods
    -------
    analytics(columns, **kwargs)
        Groups data by specified columns and calculates averages for selected metrics.
    """

    def analytics_task(self, columns, **kwargs):
        """
        Perform analytics on the specified columns and compute aggregated metrics.
        Args:
            columns (list): List of column names to group data by.
            **kwargs: Additional keyword arguments for customization.
        Returns:
            dict: A dictionary where keys are formatted strings representing group identifiers,
              and values are numpy arrays containing aggregated metrics.
        """
        # query data
        data = self.data_loader.query(columns)

        grouped = data.groupby(['age', 'sex', 'current_smoker'])

        # Convert mean values to numpy arrays if they are not already
        result = grouped.agg({
            'heart_rate': 'mean',
            'chol': 'mean',
            'blood_pressure': lambda x: self.process_blood_pressure(x).iloc[0]
        })

        # Convert the result into the desired format
        formatted_result = {}

        keys = ', heart_rate_mean, chol_mean, systolic_blood_pressure_mean, diastolic_blood_pressure_mean'
        for index, row in result.iterrows():
            age, sex, current_smoker = index
            heart_rate_mean = row['heart_rate']
            chol_mean = row['chol']
            systolic_mean = row['blood_pressure'][0]
            diastolic_mean = row['blood_pressure'][1]
            combined_key = f"age_{age}_sex_{sex}_current_smoker_{current_smoker} {keys}"
            formatted_result[combined_key] = np.array([
                heart_rate_mean, chol_mean, systolic_mean, diastolic_mean
            ])
        return formatted_result

    # Process blood pressure data
    def process_blood_pressure(self, bp_series):
        """
        Processes a series of blood pressure readings and calculates the mean
        systolic and diastolic values.
        Args:
            bp_series (pd.Series): A pandas Series containing blood pressure
                readings in the format "systolic/diastolic" (e.g., "120/80").
        Returns:
            pd.DataFrame: A DataFrame with two columns:
                - 'systolic_mean': The mean of valid systolic values, or None if no valid values exist.
                - 'diastolic_mean': The mean of valid diastolic values, or None if no valid values exist.
        Notes:
            - Invalid or non-numeric blood pressure readings are ignored.
            - If all readings are invalid, the resulting means will be None.
        """

        systolic, diastolic = zip(*bp_series.str.split('/').map(
            lambda x: (
                float(x[0]) if x[0].replace('.', '', 1).isdigit() else None,
                float(x[1]) if x[1].replace('.', '', 1).isdigit() else None
            )
        ))
        systolic = [s for s in systolic if s is not None]
        diastolic = [d for d in diastolic if d is not None]
        return pd.DataFrame({
            'systolic_mean': [sum(systolic) / len(systolic) if systolic else None],
            'diastolic_mean': [sum(diastolic) / len(diastolic) if diastolic else None]
        })
