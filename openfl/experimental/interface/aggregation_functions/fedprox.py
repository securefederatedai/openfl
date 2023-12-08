"""openfl.experimental.interface.aggregation_functions.fedprox package."""

import numpy as np

from .weighted_average import weighted_average, WeightedAverage


class FedProx(WeightedAverage):
    """Weighted average aggregation."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the function."""
        rep = "FedProx"
        return rep

    def aggregate_metrics(self, metrics, train_weights, test_weights):
        """Weighted average of loss and accuracy metrics"""
        agg_model_loss_list, agg_model_accuracy_list = metrics
        aggregated_model_training_loss = weighted_average(agg_model_loss_list, train_weights)
        aggregated_model_test_accuracy = weighted_average(agg_model_accuracy_list, test_weights)

        return (aggregated_model_training_loss, aggregated_model_test_accuracy)



 
