.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _overriding_agg_fn:

Overriding aggregation function
#######################

Usage
=====================
|productName| allows developers to use custom aggregation functions per task.
In order to do this, you should create an implementation of :class:`openfl.component.aggregation_functions.AggregationFunctionInterface`
and pass it as ``tasks.{task_name}.aggregation_type`` parameter of ``override_config`` keyword argument of :func:`openfl.native.run_experiment` native function.

.. warning::

    Currently custom aggregation functions supported in native API (``fx.run_experiment(...)``) only.
    Command Line Interface always uses FedAvg and this behavior cannot be overriden due to disability of serialization and deserialization of complex Python objects in Plan files.

``AggregationFunctionInterface`` requires a single ``call`` function.
This function receives tensors for a single parameter from multiple collaborators with additional metadata (see definition of :meth:`openfl.component.aggregation_functions.AggregationFunctionInterface.call`) and returns a single tensor that represents the result of aggregation.

.. note::
    By default, we use weighted averaging with weights equal to data parts (FedAvg).

Example
=======================

Below is an example of custom tensor clipping aggregation function that multiplies all local tensors by 0.3 and averages them according to weights equal to data parts to produce the resulting global tensor.

.. code-block:: python

    from openfl.component.aggregation_functions import AggregationFunctionInterface
    import numpy as np

    class ClippedAveraging(AggregationFunctionInterface):
        def __init__(self, ratio):
            self.ratio = ratio
            
        def call(self,
                    agg_tensor_dict,
                    weights,
                    db_iterator,
                    tensor_name,
                    fl_round,
                    *__) -> np.ndarray:
            """Aggregate tensors.

            Args:
                agg_tensor_dict: Dict of (collaborator name, tensor) pairs to aggregate.
                weights: array of floats representing data partition (sum up to 1)
                db_iterator: iterator over history of all tensors.
                    Columns: ['tensor_name', 'round', 'tags', 'nparray']
                tensor_name: name of the tensor
                fl_round: round number
                tags: tuple of tags for this tensor
            """
            clipped_tensors = []
            previous_tensor_value = None
            for record in db_iterator:
                if (
                    record['round'] == (fl_round - 1)
                    and record['tensor_name'] == tensor_name
                    and 'aggregated' in record['tags']
                ):
                    previous_tensor_value = record['nparray']
            for _, tensor in agg_tensor_dict.items():
                prev_tensor = previous_tensor_value if previous_tensor_value is not None else tensor
                delta = prev_tensor - tensor
                new_tensor = prev_tensor + delta * self.ratio
                clipped_tensors.append(new_tensor)

            return np.average(clipped_tensors, weights=weights, axis=0)

Full implementation can be found at ``openfl-tutorials/Federated_Pytorch_MNIST_custom_aggregation_Tutorial.ipynb``