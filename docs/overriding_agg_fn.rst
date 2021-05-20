.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _overriding_agg_fn:
===============================
Overriding aggregation function
===============================

-------------------------------
Usage
-------------------------------
|productName| allows developers to use custom aggregation functions per task.
In order to do this, you should:

Python API
==========

Create an implementation of :class:`openfl.component.aggregation_functions.AggregationFunctionInterface`
and pass it as ``tasks.{task_name}.aggregation_type`` parameter of ``override_config`` keyword argument of :func:`openfl.native.run_experiment` native function.

CLI
====

Choose from predefined |productName| aggregation functions:

- ``openfl.component.aggregation_functions.WeightedAverage`` (default)
- ``openfl.component.aggregation_functions.Median``
- ``openfl.component.aggregation_functions.GeometricMedian``
Or create your own implementation of :class:`openfl.component.aggregation_functions.AggregationFunctionInterface`.
After defining the aggregation behavior, you need to include it in ``plan/plan.yaml`` file of your workspace.
Inside ``tasks`` section pick a task for which you want to change the aggregation
and insert ``aggregation_type`` section with a single ``template`` key that defines a module path to your class.

Example of ``plan/plan.yaml`` with modified aggregation function:
  
.. code-block:: yaml

  # ...
  # other top-level sections
  # ...
  tasks:
    aggregated_model_validation:
      function: validate
      kwargs:
        apply: global
        metrics:
        - acc
    defaults: plan/defaults/tasks_torch.yaml
    locally_tuned_model_validation:
      function: validate
      kwargs:
      apply: local
      metrics:
      - acc
    settings: {}
    train:
      function: train_batches
      aggregation_type:
        template: openfl.component.aggregation_functions.Median  
      kwargs:
        metrics:
        - loss

``AggregationFunctionInterface`` requires a single ``call`` function.
This function receives tensors for a single parameter from multiple collaborators with additional metadata (see definition of :meth:`openfl.component.aggregation_functions.AggregationFunctionInterface.call`) and returns a single tensor that represents the result of aggregation.

.. note::
    By default, we use weighted averaging with weights equal to data parts (FedAvg).

Example
=======================

Below is an example of a custom tensor clipping aggregation function that multiplies all local tensors by 0.3 and averages them according to weights equal to data parts to produce the resulting global tensor.

.. code-block:: python

    from openfl.component.aggregation_functions import AggregationFunctionInterface
    import numpy as np

    class ClippedAveraging(AggregationFunctionInterface):
        def __init__(self, ratio):
            self.ratio = ratio
            
        def call(self,
                local_tensors,
                db_iterator,
                tensor_name,
                fl_round,
                *__):
            """Aggregate tensors.

            Args:
                local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
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
                    and 'delta' not in record['tags']
                ):
                    previous_tensor_value = record['nparray']
            weights = []
            for local_tensor in local_tensors:
                prev_tensor = previous_tensor_value if previous_tensor_value is not None else local_tensor.tensor
                delta = local_tensor.tensor - prev_tensor
                new_tensor = prev_tensor + delta * self.ratio
                clipped_tensors.append(new_tensor)
                weights.append(local_tensor.weight)

            return np.average(clipped_tensors, weights=weights, axis=0)

Full implementation can be found at ``openfl-tutorials/Federated_Pytorch_MNIST_custom_aggregation_Tutorial.ipynb``