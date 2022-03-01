.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _overriding_agg_fn:

*****************************
Override Aggregation Function
*****************************

With the aggregator-based workflow, you can use custom aggregation functions for each task via Python\*\  API or command line interface.


Python API
==========

1. Create an implementation of :class:`openfl.component.aggregation_functions.core.AggregationFunction`.

2. In the ``override_config`` keyword argument of the :func:`openfl.native.run_experiment` native function, pass the implementation as a ``tasks.{task_name}.aggregation_type`` parameter.

.. note::
    See `Federated PyTorch MNIST Tutorial <https://github.com/intel/openfl/blob/develop/openfl-tutorials/Federated_Pytorch_MNIST_custom_aggregation_Tutorial.ipynb>`_ for an example of the custom aggregation function.
    

Command Line Interface
======================

Predefined Aggregation Functions
--------------------------------

Choose from the following predefined aggregation functions:

- ``openfl.component.aggregation_functions.WeightedAverage`` (default)
- ``openfl.component.aggregation_functions.Median``
- ``openfl.component.aggregation_functions.GeometricMedian``
- ``openfl.component.aggregation_functions.AdagradAdaptiveAggregation``
- ``openfl.component.aggregation_functions.AdamAdaptiveAggregation``
- ``openfl.component.aggregation_functions.YogiAdaptiveAggregation``


.. note::
    To create adaptive aggregation functions,
    the user must specify parameters for the aggregation optimizer
    (``NumPyAdagrad``, ``NumPyAdam`` or ``NumPyYogi``) that will aggregate
    the global model. Theese parameters parameters are passed via **keywords**.

    Also, user must pass one of the arguments: ``params``
    - model parameters (a dictionary with named model parameters
    in the form of numpy arrays), or pass ``model_interface``
    - an instance of the `ModelInterface <https://github.com/intel/openfl/blob/develop/openfl/interface/interactive_api/experiment.py>`_ class.
    If user pass both ``params`` and ``model_interface``,
    then the optimizer parameters are initialized via
    ``params``, ignoring ``model_interface`` argument.

    See the `AdagradAdaptiveAggregation
    <https://github.com/intel/openfl/blob/develop/openfl/component/aggregation_functions/adagrad_adaptive_aggregation.py>`_
    definitions for details.

    `Adaptive federated optimization <https://arxiv.org/pdf/2003.00295.pdf>`_ original paper.

``AdagradAdaptiveAggregation`` usage example:

.. code-block:: python

    from openfl.interface.interactive_api.experiment import TaskInterface, ModelInterface
    from openfl.component.aggregation_functions import AdagradAdaptiveAggregation

    TI = TaskInterface()
    MI = ModelInterface(model=model,
                        optimizer=optimizer,
                        framework_plugin=framework_adapter)
    ...

    # Creating aggregation function
    agg_fn = AdagradAdaptiveAggregation(model_interface=MI,
                                        learning_rate=0.4)

    # Define training task
    @TI.register_fl_task(model='model', data_loader='train_loader', \
                            device='device', optimizer='optimizer')
    @TI.set_aggregation_function(agg_fn)
    def train(...):
    ...

You can define your own numpy based optimizer,
which will be used for global model aggreagation:

.. code-block:: python

    from openfl.utilities.optimizers.numpy.base_optimizer import Optimizer

    class MyOpt(Optimizer):
        """My optimizer implementation."""

        def __init__(
            self,
            *,
            params: Optional[Dict[str, np.ndarray]] = None,
            model_interface=None,
            learning_rate: float = 0.001,
            param1: Any = None,
            param2: Any = None
        ) -> None:
            """Initialize.

            Args:
                params: Parameters to be stored for optimization.
                model_interface: Model interface instance to provide parameters.
                learning_rate: Tuning parameter that determines
                    the step size at each iteration.
                param1: My own defined parameter.
                param2: My own defined parameter.
            """
            super().__init__()
            pass # Your code here!

        def step(self, gradients: Dict[str, np.ndarray]) -> None:
            """
            Perform a single step for parameter update.

            Implement your own optimizer weights update rule.

            Args:
                gradients: Partial derivatives with respect to optimized parameters.
            """
            pass # Your code here!
    ...

    from openfl.component.aggregation_functions import WeightedAverage
    from openfl.component.aggregation_functions.core import AdaptiveAggregation

    # Creating your implemented optimizer instance based on numpy:
    my_own_optimizer = MyOpt(model_interface=MI, learning_rate=0.01)

    # Creating aggregation function
    agg_fn = AdaptiveAggregation(optimizer=my_own_optimizer,
                                 agg_func=WeightedAverage()) # WeightedAverage() is used for aggregating
                                                             # parameters that are not inside the given optimizer.

    # Define training task
    @TI.register_fl_task(model='model', data_loader='train_loader', \
                            device='device', optimizer='optimizer')
    @TI.set_aggregation_function(agg_fn)
    def train(...):
    ...

.. note::
    If you do not understand how to write your own numpy based optimizer, please see the `NumPyAdagrad <https://github.com/intel/openfl/blob/develop/openfl/utilities/optimizers/numpy/adagrad_optimizer.py>`_ and
    `AdaptiveAggregation <https://github.com/intel/openfl/blob/develop/openfl/component/aggregation_functions/core/adaptive_aggregation.py>`_ definitions for details.

Custom Aggregation Functions
----------------------------

You can also create your own implementation of :class:`openfl.component.aggregation_functions.core.AggregationFunction`. See `example <https://github.com/intel/openfl/blob/develop/openfl-tutorials/Federated_Pytorch_MNIST_custom_aggregation_Tutorial.ipynb>`_ for details.

1. Define the behavior of the aggregation.

2. Include the implementation in the **plan.yaml** file in the **plan** directory of your workspace.

3. In the **tasks** section,  pick a task for which you want to change the aggregation and insert ``aggregation_type`` section with a single ``template`` key that defines a module path to your class.

The following is an example of a **plan.yaml** with a modified aggregation function:
  
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


Interactive API
================
You can override aggregation function that will be used for the task this function corresponds to.
In order to do this, call the ``set_aggregation_function`` decorator method of ``TaskInterface`` and pass ``AggregationFunction`` subclass instance as a parameter.
For example, you can try:

.. code-block:: python

    from openfl.component.aggregation_functions import Median
    TI = TaskInterface()
    agg_fn = Median()
    @TI.register_fl_task(model='model', data_loader='train_loader', \
                         device='device', optimizer='optimizer')
    @TI.set_aggregation_function(agg_fn)

.. warning::
    All tasks with the same type of aggregation use the same class instance.
    If ``AggregationFunction`` implementation has its own state, then this state will be shared across tasks.


``AggregationFunction`` requires a single ``call`` function.
This function receives tensors for a single parameter from multiple collaborators with additional metadata (see definition of :meth:`openfl.component.aggregation_functions.core.AggregationFunction.call`) and returns a single tensor that represents the result of aggregation.


.. note::
    See the `definition <https://github.com/intel/openfl/blob/develop/openfl/component/aggregation_functions/core/interface.py>`_ of :class:`openfl.component.aggregation_functions.core.AggregationFunction.call` for details.


Example of a Custom Aggregation Function
========================================

This is an example of a custom tensor clipping aggregation function that multiplies all local tensors by 0.3 and averages them according to weights equal to data parts to produce the resulting global tensor.

.. code-block:: python

    from openfl.component.aggregation_functions import AggregationFunction
    import numpy as np

    class ClippedAveraging(AggregationFunction):
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
                db_iterator: iterator over history of all tensors. Columns:
                    - 'tensor_name': name of the tensor.
                        Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                    - 'round': 0-based number of round corresponding to this tensor.
                    - 'tags': tuple of tensor tags. Tags that can appear:
                        - 'model' indicates that the tensor is a model parameter.
                        - 'trained' indicates that tensor is a part of a training result.
                            These tensors are passed to the aggregator node after local learning.
                        - 'aggregated' indicates that tensor is a result of aggregation.
                            These tensors are sent to collaborators for the next round.
                        - 'delta' indicates that value is a difference between rounds
                            for a specific tensor.
                        also one of the tags is a collaborator name
                        if it corresponds to a result of a local task.

                    - 'nparray': value of the tensor.
                tensor_name: name of the tensor
                fl_round: round number
                tags: tuple of tags for this tensor
            Returns:
                np.ndarray: aggregated tensor
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

A full implementation can be found at `Federated_Pytorch_MNIST_custom_aggregation_Tutorial.ipynb <https://github.com/intel/openfl/blob/develop/openfl-tutorials/Federated_Pytorch_MNIST_custom_aggregation_Tutorial.ipynb>`_
