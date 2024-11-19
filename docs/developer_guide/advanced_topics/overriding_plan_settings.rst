.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _overriding_plan_settings:

***********************
Updating plan settings
***********************

With the director-based workflow, you can use custom plan settings before starting the experiment. Changing plan settings in command line interface is straightforward by modifying plan.yaml.
When using Python API or Director Envoy based interactive API (Deprecated), **override_config** can be used to update plan settings. 


Python API
==========

Modify the plan settings:

.. code-block:: python

    final_fl_model = fx.run_experiment(collaborators, override_config={
    'aggregator.settings.rounds_to_train': 5,
    'aggregator.settings.log_metric_callback': write_metric,
    })


Director Envoy Based Interactive API Interface (Deprecated)
===========================================================
Once you create an FL_experiment object, a basic federated learning plan with default settings is created. To check the default plan settings, print the plan as shown below:

.. code-block:: python

    fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name)
    import openfl.native as fx
    print(fx.get_plan(fl_plan=fl_experiment.plan))

Here is an example of the default plan settings that get displayed:

.. code-block:: python

    "aggregator.settings.best_state_path": "save/best.pbuf",
    "aggregator.settings.db_store_rounds": 2,
    "aggregator.settings.init_state_path": "save/init.pbuf",
    "aggregator.settings.last_state_path": "save/last.pbuf",
    "aggregator.settings.rounds_to_train": 10,
    "aggregator.settings.write_logs": true,
    "aggregator.template": "openfl.component.Aggregator",
    "assigner.settings.task_groups.0.name": "train_and_validate",
    "assigner.settings.task_groups.0.percentage": 1.0,
    "assigner.settings.task_groups.0.tasks.0": "aggregated_model_validation",
    "assigner.settings.task_groups.0.tasks.1": "train",
    "assigner.settings.task_groups.0.tasks.2": "locally_tuned_model_validation",
    "assigner.template": "openfl.component.RandomGroupedAssigner",
    "collaborator.settings.db_store_rounds": 1,
    "collaborator.settings.delta_updates": false,
    "collaborator.settings.opt_treatment": "RESET",
    "collaborator.template": "openfl.component.Collaborator",
    "compression_pipeline.settings": {},
    "compression_pipeline.template": "openfl.pipelines.NoCompressionPipeline",
    "data_loader.settings": {},
    "data_loader.template": "openfl.federated.DataLoader",
    "network.settings.agg_addr": "auto",
    "network.settings.agg_port": "auto",
    "network.settings.cert_folder": "cert",
    "network.settings.client_reconnect_interval": 5,
    "network.settings.disable_client_auth": false,
    "network.settings.hash_salt": "auto",
    "network.settings.tls": true,
    "network.template": "openfl.federation.Network",
    "task_runner.settings": {},
    "task_runner.template": "openfl.federated.task.task_runner.CoreTaskRunner",
    "tasks.settings": {}


Use **override_config** with FL_experiment.start to make any changes to the default plan settings. It's essentially a dictionary with the keys corresponding to plan parameters along with the corresponding values (or list of values). Any new key entry will be added to the plan and for any existing key present in the plan, the value will be overrriden.


.. code-block:: python

    fl_experiment.start(model_provider=MI, 
                   task_keeper=TI,
                   data_loader=fed_dataset,
                   rounds_to_train=5,
                   opt_treatment='CONTINUE_GLOBAL',
                   override_config={'aggregator.settings.db_store_rounds': 1, 'compression_pipeline.template': 'openfl.pipelines.KCPipeline', 'compression_pipeline.settings.n_clusters': 2})


Since 'aggregator.settings.db_store_rounds' and 'compression_pipeline.template' fields are already present in the plan, the values of these fields get replaced. Field  'compression_pipeline.settings.n_clusters' is a new entry that gets added to the plan:

.. code-block:: python

    INFO     Updating aggregator.settings.db_store_rounds to 1...                                                                           native.py:102

    INFO     Updating compression_pipeline.template to openfl.pipelines.KCPipeline...                                                       native.py:102

    INFO     Did not find compression_pipeline.settings.n_clusters in config. Make sure it should exist. Creating...                        native.py:105


A full implementation can be found at `Federated_Pytorch_MNIST_Tutorial.ipynb <https://github.com/intel/openfl/blob/develop/openfl-tutorials/Federated_Pytorch_MNIST_Tutorial.ipynb>`_ and at `Tensorflow_MNIST.ipynb <https://github.com/securefederatedai/openfl/tree/main/openfl-tutorials/deprecated/interactive_api/Tensorflow_MNIST/workspace/Tensorflow_MNIST.ipynb>`_.
