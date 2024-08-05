# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Adagrad adaptive aggregation module."""

from typing import Dict, Optional

import numpy as np

from openfl.interface.aggregation_functions.core import AdaptiveAggregation, AggregationFunction
from openfl.interface.aggregation_functions.weighted_average import WeightedAverage
from openfl.utilities.optimizers.numpy import NumPyAdagrad

DEFAULT_AGG_FUNC = WeightedAverage()


class AdagradAdaptiveAggregation(AdaptiveAggregation):
    """Adagrad adaptive Federated Aggregation funtcion."""

    def __init__(
        self,
        *,
        agg_func: AggregationFunction = DEFAULT_AGG_FUNC,
        params: Optional[Dict[str, np.ndarray]] = None,
        model_interface=None,
        learning_rate: float = 0.01,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-10,
    ) -> None:
        """Initialize the AdagradAdaptiveAggregation object.

        Args:
            agg_func (AggregationFunction): Aggregate function for aggregating
                parameters that are not inside the optimizer (default:
                WeightedAverage()).
            params (Optional[Dict[str, np.ndarray]]): Parameters to be stored
                for optimization.
            model_interface: Model interface instance to provide parameters.
            learning_rate (float): Tuning parameter that determines
                the step size at each iteration.
            initial_accumulator_value (float): Initial value for squared
                gradients.
            epsilon (float): Value for computational stability.
        """
        opt = NumPyAdagrad(
            params=params,
            model_interface=model_interface,
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon,
        )
        super().__init__(opt, agg_func)
