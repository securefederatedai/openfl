import numpy as np
from openfl.interface.aggregation_functions.core import AggregationFunction


class AggregateHealthMetrics(AggregationFunction):
    """Aggregation logic for Smokers Health analytics."""

    def call(self, local_tensors, *_) -> dict:
        """
        Aggregates local metrics into global averages.

        Parameters
        ----------
        local_tensors : list
            A list of dictionaries containing local metrics.

        Returns
        -------
        dict
            A dictionary with global averages for each metric.
        """
        if not local_tensors:
            raise ValueError("No local metrics to aggregate.")
        
        agg_histogram = np.zeros_like(local_tensors[0].tensor)
        for local_tensor in local_tensors:
            agg_histogram += local_tensor.tensor / len(local_tensors)
        return agg_histogram