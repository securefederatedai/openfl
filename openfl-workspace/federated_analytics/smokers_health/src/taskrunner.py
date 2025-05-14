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
            combined_key = f"{age}_{sex}_current_smoker_{current_smoker} {keys}"
            formatted_result[combined_key] = np.array([
                heart_rate_mean, chol_mean, systolic_mean, diastolic_mean
            ])
        return formatted_result

    # Process blood pressure data
    def process_blood_pressure(self, bp_series):
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
