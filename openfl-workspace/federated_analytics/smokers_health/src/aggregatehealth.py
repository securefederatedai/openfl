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
        
        print("type(local_tensors):", type(local_tensors))
        print("len(local_tensors):", len(local_tensors))
        print("local_tensors[0]:", local_tensors[0])
        print("type(local_tensors[0]):", type(local_tensors[0]))
        print("len(local_tensors[0]):", len(local_tensors[0]))
        print("local_tensors[0].keys():", local_tensors[0].keys())
        print("local_tensors[0].values():", local_tensors[0].values())
        print("local_tensors:", local_tensors)
        aggregated = {}
        for key in local_tensors[0].keys():
            values = [tensor[key] for tensor in local_tensors]
            aggregated[key] = np.mean(values)

        return aggregated
