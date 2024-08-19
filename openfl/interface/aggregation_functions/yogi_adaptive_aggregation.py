# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Yogi adaptive aggregation module."""

from typing import Dict, Optional, Tuple

import numpy as np

from openfl.interface.aggregation_functions.core import AdaptiveAggregation, AggregationFunction
from openfl.interface.aggregation_functions.weighted_average import WeightedAverage
from openfl.utilities.optimizers.numpy import NumPyYogi

DEFAULT_AGG_FUNC = WeightedAverage()


class YogiAdaptiveAggregation(AdaptiveAggregation):
    """Yogi adaptive Federated Aggregation funtcion."""

    def __init__(
        self,
        *,
        agg_func: AggregationFunction = DEFAULT_AGG_FUNC,
        params: Optional[Dict[str, np.ndarray]] = None,
        model_interface=None,
        learning_rate: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        initial_accumulator_value: float = 0.0,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize the YogiAdaptiveAggregation object.

        Args:
            agg_func (AggregationFunction): Aggregate function for aggregating
                parameters that are not inside the optimizer (default:
                WeightedAverage()).
            params (Optional[Dict[str, np.ndarray]]): Parameters to be stored
                for optimization.
            model_interface: Model interface instance to provide parameters.
            learning_rate (float): Tuning parameter that determines
                the step size at each iteration.
            betas (Tuple[float, float]): Coefficients used for computing
                running averages of gradient and its square.
            initial_accumulator_value (float): Initial value for gradients
                and squared gradients.
            epsilon (float): Value for computational stability.
        """
        opt = NumPyYogi(
            params=params,
            model_interface=model_interface,
            learning_rate=learning_rate,
            betas=betas,
            initial_accumulator_value=initial_accumulator_value,
            epsilo=epsilon,
        )
        super().__init__(opt, agg_func)
