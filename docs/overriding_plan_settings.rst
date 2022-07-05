.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _overriding_plan_settings:

***********************
Updating plan settings
***********************

With the director-based workflow, you can use custom plan settings before starting the experiment. Changing plan settings in command line interface is straightforward by modifying plan.yaml.
When using Python API or Director Envoy based interactive API, **override_config** can be used to update plan settings. 


Python API
==========

Modify the plan settings:

.. code-block:: python

    final_fl_model = fx.run_experiment(collaborators, override_config={
    'aggregator.settings.rounds_to_train': 5,
    'aggregator.settings.log_metric_callback': write_metric,
    })


Director Envoy Based Interactive API Interface
==============================================
Use **override_config** with FL_experiment.start:

.. code-block:: python

    import openfl.native as fx
    print(fx.get_plan(plan_config_path='plan/plan.yaml'))

    fl_experiment.start(model_provider=MI, 
                   task_keeper=TI,
                   data_loader=fed_dataset,
                   rounds_to_train=5,
                   opt_treatment='CONTINUE_GLOBAL',
                   override_config={'compression_pipeline.template': 'openfl.pipelines.KCPipeline', 'compression_pipeline.settings.n_clusters': 2})


A full implementation can be found at `Federated_Pytorch_MNIST_Tutorial.ipynb <https://github.com/intel/openfl/blob/develop/openfl-tutorials/Federated_Pytorch_MNIST_Tutorial.ipynb>`_ and at `Tensorflow_MNIST.ipynb <https://github.com/intel/openfl/blob/develop/openfl-tutorials/interactive_api/Tensorflow_MNIST/workspace/Tensorflow_MNIST.ipynb>`_.