# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openfl.interface.aggregation_functions.core import AggregationFunction


class AggregateHealthMetrics(AggregationFunction):
    """Aggregation logic for Smokers Health analytics."""

    def call(self, local_tensors, *_) -> dict:
        """
        Aggregates local tensors which contains mean of local health metrics such as
        heart_rate_mean, cholesterol, systolic_blood_pressure, and
        diastolic_blood_pressure which are grouped by age, sex and if they smoke or not.
        Each tensor represents local metrics for these health parameters.

        Args:
            local_tensors (list): A list of objects, each containing a `tensor` attribute
                      that represents local means for the health metrics.
            *_: Additional arguments (unused).
        Returns:
            dict: A dictionary containing the aggregated means for each health metric.
        Raises:
            ValueError: If the input list `local_tensors` is empty, indicating
            that there are no metrics to aggregate.
        """

        if not local_tensors:
            raise ValueError("No local metrics to aggregate.")

        agg_histogram = np.zeros_like(local_tensors[0].tensor)
        for local_tensor in local_tensors:
            agg_histogram += local_tensor.tensor / len(local_tensors)
        return agg_histogram
