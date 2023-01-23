.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _log_metric_callback:

***********************
Metric Logging Callback
***********************
By default, both the director based flow and the taskrunner API support `Tensorboard <https://www.tensorflow.org/tensorboard/get_started>`_ to log metrics.
Once the experiment is over, the logs can be invoked from the workspace with :code:`tensorboard --logdir logs`. The metrics that are logged by default are:

- Aggregated model validation accuracy (Aggregator/aggregated_model_validate/acc, validate_agg/aggregated_model_validate/acc)
- Locally tuned model validation accuracy (Aggregator/locally_tuned_model_validate/acc, validate_local/locally_tuned_model_validate/acc)
- Train loss (Aggregator/train/train_loss, trained/train/train_loss)

You can also use custom metric logging function for each task via Python\*\  API or command line interface. This function calls on the aggregator node.

Python API
==========

For logging metrics through Tensorboard, once :code:`fl_experiment.stream_metrics()` is called from the frontend API, it saves logs in the tensorboard format.
After the experiment has finished, the logs can be invoked from the workspace with :code:`tensorboard --logdir logs`. 

You could also add your custom metric logging function by defining the function with the follow signature:

.. code-block:: python

    def callback_name(node_name, task_name, metric_name, metric, round_number):
        """
        Write metric callback 

        Args:
            node_name (str): Name of node, which generate metric 
            task_name (str): Name of task
            metric_name (str): Name of metric 
            metric (np.ndarray): Metric value
            round_number (int): Round number
        """
        your code 


Example of a Metric Callback
============================

This example shows how to use MLFlow logger to log metrics:

.. code-block:: python

    import mlflow

    def callback_name(node_name, task_name, metric_name, metric, round_number):
        """
        Write metric callback 

        Args:
            node_name (str): Name of node, which generate metric 
            task_name (str): Name of task
            metric_name (str): Name of metric 
            metric (np.ndarray): Metric value
            round_number (int): Round number
        """
        mlflow.log_metrics({f'{node_name}/{task_name}/{metric_name}': float(metric), 'round_number': round_number})

You could view the log results either through UI interactively by typing :code:`mlflow ui` or through the use of :code:`MLflowClient`. By default, only the last logged value of the metric is returned. 
If you want to retrieve all the values of a given metric, uses :code:`mlflow.get_metric_history` method.

.. code-block:: python

    import mlflow
    client = mlflow.tracking.MlflowClient()
    print(client.get_metric_history("<RUN ID>", "validate_local/locally_tuned_model_validation/accuracy"))

  
Command Line Interface
======================

For logging through Tensorboard, enable the parameter :code:`write_logs : true` in `aggregator's plan settings <https://github.com/intel/openfl/blob/develop/openfl-workspace/workspace/plan/defaults/aggregator.yaml>`_ :

.. code-block:: yaml

  aggregator :
    template : openfl.component.Aggregator
    settings :
        write_logs : true

Follow the steps below to write your custom callback function instead. As an example, a full implementation can be found at `Federated_Pytorch_MNIST_Tutorial.ipynb <https://github.com/intel/openfl/blob/develop/openfl-tutorials/Federated_Pytorch_MNIST_Tutorial.ipynb>`_ and in the **torch_cnn_mnist** workspace.

1. Define the callback function, like how you defined in Python API, in the **src** directory in your workspace.

2. Provide a way to your function with the ``log_metric_callback`` key in the ``aggregator`` section of the **plan.yaml** file in your workspace. 

.. code-block:: yaml

  aggregator :
    defaults : plan/defaults/aggregator.yaml
    template : openfl.component.Aggregator
    settings :
      init_state_path     : save/torch_cnn_mnist_init.pbuf
      best_state_path     : save/torch_cnn_mnist_best.pbuf
      last_state_path     : save/torch_cnn_mnist_last.pbuf
      rounds_to_train     : 10
      write_logs          : true
      log_metric_callback :
        template : src.mnist_utils.callback_name


Example of a Metric Callback
============================

The following is an example of a log metric callback, which writes metric values to the TensorBoard.

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter('./logs/cnn_mnist', flush_secs=5)


    def write_metric(node_name, task_name, metric_name, metric, round_number):
        writer.add_scalar("{}/{}/{}".format(node_name, task_name, metric_name),
                        metric, round_number) 
