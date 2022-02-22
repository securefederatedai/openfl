# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Yogi adaptive aggregation module."""


from openfl.utilities.optimizers.numpy import NumPyYogi
from .core import AdaptiveAggregation
from .core import AggregationFunction
from .weighted_average import WeightedAverage


DEFAULT_AGG_FUNC = WeightedAverage()


class YogiAdaptiveAggregation(AdaptiveAggregation):
    """Yogi adaptive Federated Aggregation funtcion."""

    def __init__(
        self,
        default_agg_func: AggregationFunction = DEFAULT_AGG_FUNC,
        **kwargs
    ) -> None:
        """Initialize.

        Args:
            default_agg_func: Aggregate function for aggregating
                parameters that are not inside the optimizer (default: WeightedAverage()).
            kwargs: parameters to NumPyYogi optimizer. More details in the
                openfl.utilities.optimizers.numpy.adagrad_optimizer.NumPyYogi module.
        """
        opt = NumPyYogi(**kwargs)
        super().__init__(opt, default_agg_func)
