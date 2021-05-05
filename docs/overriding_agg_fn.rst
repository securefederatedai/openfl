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

    class ClippedAveraging(AggregationFunctionInterface): # interface implementation
        def clip(self, tensor):
            # clip could be any number of clipping functions
            return tensor * .3
        def call(self,
                    tensors,
                    weights,
                    db_iterator,
                    _,
                    fl_round,
                    *__):
            """Aggregate tensors.

            Args:
                tensors: array of `np.ndarray`s of tensors to aggregate.
                weights: array of floats representing data partition (sum up to 1)
                db_iterator: iterator over history of aggregated versions of this tensor
                tensor_name: name of the tensor
                fl_round: round number
                tags: tuple of tags for this tensor
            """
            clipped_tensors = []
            previous_tensor_value = None
            for z in db_iterator: # here we use an iterator to find previous value of this tensor
                if z['round'] == (fl_round - 1):
                    previous_tensor_value = z['nparray']
            for tensor in tensors:
                _previous_tensor_value = previous_tensor_value if previous_tensor_value is not None else tensor
                delta = tensor - _previous_tensor_value
                clipped_tensor = _previous_tensor_value + self.clip(delta) # clip the update 
                clipped_tensors.append(clipped_tensor)

            return np.average(clipped_tensors, weights=weights, axis=0) # average of clipped updates

Full implementation can be found at ``openfl-tutorials/Federated_Pytorch_MNIST_custom_aggregation_Tutorial.ipynb``