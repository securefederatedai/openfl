.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _log_metric_callback:

***********************
Metric Logging Callback
***********************

With the director-based workflow, you can use custom metric logging function for each task via Python\*\  API or command line interface. This function calls on the aggregator node.


Python API
==========

Define the function with the follow signature:

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

Command Line Interface
======================

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


A full implementation can be found at `Federated_Pytorch_MNIST_Tutorial.ipynb <https://github.com/intel/openfl/blob/develop/openfl-tutorials/Federated_Pytorch_MNIST_Tutorial.ipynb>`_ and in the **torch_cnn_mnist** workspace.