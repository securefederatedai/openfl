.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _overriding_agg_fn:

Overriding aggregation function
#######################

|productName| allows developers to use custom aggregation functions per task.
In order to do this, you should create an implementation of :class:`openfl.component.aggregation_functions.AggregationFunctionInterface` and pass it as`tasks.{task_name}.aggregation_type` parameter of `override_config` keyword argument of run_experiment native function.


This interface requires a single `call` function.
This function receives tensors for a single parameter from multiple collaborators with additional metadata (see definition of :meth:`openfl.component.aggregation_functions.AggregationFunctionInterface.call`) and returns a single tensor that represents the result of aggregation.

.. note::
    By default, we use weighted averaging with weights equal to data parts (FedAvg).
