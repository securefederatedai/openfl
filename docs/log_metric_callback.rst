.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _log_metric_callback:
===============================
Metric logging callback
===============================

-------------------------------
Usage
-------------------------------
|productName| allows developers to use custom metric logging functions. This function will call on aggregator node.
In order to define such function, you should:

Python API
==========
Define function with follow signature:

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
CLI
====

Define callback function similar way like in python api in ``src`` folder of your workspace. And provide a way to your function in ``aggregator`` part of ``plan/plan.yaml`` file in your workspace, use ``log_metric_callback`` key: 

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



Example
=======================

Below is an example of a log metric callback, which writes metric values to tensorboard

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter('./logs/cnn_mnist', flush_secs=5)


    def write_metric(node_name, task_name, metric_name, metric, round_number):
        writer.add_scalar("{}/{}/{}".format(node_name, task_name, metric_name),
                        metric, round_number) 

Full implementation can be found in ``openfl-tutorials/Federated_Pytorch_MNIST_Tutorial.ipynb`` and in ``torch_cnn_mnist`` workspace