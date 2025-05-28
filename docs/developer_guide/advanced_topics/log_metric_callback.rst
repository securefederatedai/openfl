.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _log_metric_callback:

**************
Metric Logging
**************
TaskRunner API supports a built-in callback to log metrics to a plain text file, or in a `TensorBoard <https://www.tensorflow.org/tensorboard/get_started>`_ compatible format.
To enable metric logging, you need to set the :code:`write_logs` parameter in the plan settings of the aggregator component to :code:`true`. An example of the plan settings is shown below:

.. code-block:: yaml

  aggregator :
    template : openfl.component.Aggregator
    settings :
        write_logs : true

Metrics are captured at the end of each round and written to a file in the format of :code:`<node_name>/<task_name>/<metric_name>`. The metric values are written in a plain text file, which can be used for further analysis or visualization. These logs are written under :code:`logs/`.

Example contents of the log file:

.. code-block:: text
    {"round_number": 0, "elapsed_seconds": 11.330059889999973, "aggregator/locally_tuned_model_validation/accuracy": 0.9625962972640991, "aggregator/aggregated_model_validation/accuracy": 0.13151314854621887, "aggregator/train/loss": 0.24131347239017487}
    {"round_number": 1, "elapsed_seconds": 11.558851689999983, "aggregator/locally_tuned_model_validation/accuracy": 0.967796802520752, "aggregator/aggregated_model_validation/accuracy": 0.9674967527389526, "aggregator/train/loss": 0.08967384696006775}

To log metrics for visualization on TensorBoard, set the environment variable :code:`TENSORBOARD=1` before starting the aggregator/collaborator. Note that this still requires :code:`write_logs` to be set to :code:`true` in the plan settings as shown above.

Summaries are written under :code:`logs/tensorboard/`. To visualize the logs, run the following command in a separate shell:

.. code-block:: bash

    tensorboard --logdir logs/tensorboard/

You may use a compatible browser and navigate to the provided URL to open the TensorBoard dashboard.

Example of MLFlow's Metric Callback
=====================================

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

Known issues
============

Metric writing via TensorBoard is not supported within enclaves due to lack of full support for pythonic multiprocessing within Gramine. 
By default, metrics are only synchronously written to a text file when enabled. Outside enclave environments, you may enable tensorboard logging via :code:`TENSORBOARD=1` environment variable. We are assessing ways to synchronously write tensorboard-compatible proto files. If this is a feature you are interested in, or would like to contribute a PR, please create an issue or a pull request.
