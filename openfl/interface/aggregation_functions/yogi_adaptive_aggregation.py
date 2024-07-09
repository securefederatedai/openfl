# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Yogi adaptive aggregation module."""

from typing import Dict, Optional, Tuple

import numpy as np

from openfl.interface.aggregation_functions.core import (
    AdaptiveAggregation,
    AggregationFunction,
)
from openfl.interface.aggregation_functions.weighted_average import (
    WeightedAverage,
)
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
        """Initialize.

        Args:
            agg_func: Aggregate function for aggregating
                parameters that are not inside the optimizer (default: WeightedAverage()).
            params: Parameters to be stored for optimization.
            model_interface: Model interface instance to provide parameters.
            learning_rate: Tuning parameter that determines
                the step size at each iteration.
            betas: Coefficients used for computing running
                averages of gradient and its square.
            initial_accumulator_value: Initial value for gradients
                and squared gradients.
            epsilon: Value for computational stability.
        """
        opt = NumPyYogi(params=params,
                        model_interface=model_interface,
                        learning_rate=learning_rate,
                        betas=betas,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilo=epsilon)
        super().__init__(opt, agg_func)
