.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _interactive_api:

*******************************************
Interactive Python API (Beta)
*******************************************

The Open Federated Learning (|productName|) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\  notebook or a Python script. 

    - :ref:`federation_api_prerequisites`
    - :ref:`federation_api_define_fl_experiment`
    - :ref:`federation_api_start_fl_experiment`
    - :ref:`federation_api_observe_fl_experiment`
    - :ref:`federation_api_complete_fl_experiment`

.. _federation_api_prerequisites:

Prerequisites
=============

The Experiment manager requires the following:

Python Intepreter
    Create a virtual Python environment with packages required for conducting the experiment. The Python environment is replicated on collaborator nodes.

A Local Experiment Workspace
    Initialize a workspace by creating an empty directory and placing inside the workspace a Jupyter\*\  notebook or a Python script.
    
    Items in the workspace may include:
    
        - source code of objects imported into the notebook from local modules
        - local test data stored in a **data** directory
        - certificates stored in a **cert** directory
        
    .. note::
    
        This workspace will be archived and transferred to collaborator nodes. Ensure only relevant source code or resources are stored in the workspace.
         **data** and **cert** directories will not be included in the archive.


.. _federation_api_define_fl_experiment:

Define a Federated Learning Experiment
======================================

The definition process of a federated learning experiment uses the interactive Python API to set up several interface entities and experiment parameters.

The following are the interactive Python API to define an experiment:

    - :ref:`federation_api`
    - :ref:`experiment_api`
    
        - :ref:`experiment_api_modelinterface`
        - :ref:`experiment_api_taskinterface`
        - :ref:`experiment_api_datainterface`
    

.. note::
    Each federation is bound to some Machine Learning problem in a sense that all collaborators dataset shards should allow to solve the same data science problem. 
    For example object detection and semantic segmentation problems should be solved in different federations. \


.. _federation_api:

Federation API
--------------

The *Federation* entity is designed to be a bridge between a notebook and *Director*.


1. Import the Federation class from openfl package

    .. code-block:: python

        from openfl.interface.interactive_api.federation import Federation


2. Initialize the Federation object with the Director node network address and encryption settings.

    .. code-block:: python

        federation = Federation(
            client_id: str, director_node_fqdn: str, director_port: str
            tls: bool, cert_chain: str, api_cert: str, api_private_key: str)

    .. note::
        You may disable mTLS in trusted environments or enable mTLS by providing paths to the certificate chain of the API authority, aggregator certificate, and a private key.


.. note::
    Methods available in the Federation API:
        
        - :code:`get_dummy_shard_descriptor`: creates a dummy shard descriptor for debugging the experiment pipeline
        - :code:`get_shard_registry`: returns information about the Envoys connected to the Director and their shard descriptors

.. _experiment_api:

Experiment API
----------------

The *Experiment* entity registers training-related objects, federated learning (FL) tasks, and settings.

1. Import the FLExperiment class from openfl package

    .. code-block:: python

        from openfl.interface.interactive_api.experiment import FLExperiment

2. Initialize the experiment with the following parameters: a federation object and a unique experiment name.

    .. code-block:: python

        fl_experiment = FLExperiment(federation: Federation, experiment_name: str)

3. Import these supplementary interface classes: :code:`TaskInterface`, :code:`DataInterface`, and :code:`ModelInterface`.

    .. code-block:: python

        from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface


.. _experiment_api_modelinterface:

Register the Model and Optimizer ( :code:`ModelInterface` )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instantiate and initialize a model and optimizer in your preferred deep learning framework.

    .. code-block:: python

        from openfl.interface.interactive_api.experiment import ModelInterface
        MI = ModelInterface(model, optimizer, framework_plugin: str)
    
The initialized model and optimizer objects should be passed to the :code:`ModelInterface` along with the path to correct Framework Adapter plugin inside the |productName| package 
or from local workspace.

.. note::
    The |productName| interactive API supports *TensorFlow* and *PyTorch* frameworks via existing plugins. 
    User can add support for other deep learning frameworks via the plugin interface and point to your implementation of a :code:`framework_plugin` in :code:`ModelInterface`. 


.. _experiment_api_taskinterface:

Register FL Tasks ( :code:`TaskInterface` )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An FL task accepts the following objects:

    - :code:`model` - will be rebuilt with relevant weights for every task by `TaskRunner`
    - :code:`data_loader` - data loader that will provide local data
    - :code:`device` - a device to be used for execution on collaborator machines
    - :code:`optimizer` (optional) - model optimizer; only for training tasks

Register an FL task and accompanying information. 

    .. code-block:: python

        TI = TaskInterface()

        task_settings = {
            'batch_size': 32,
            'some_arg': 228,
        }
        @TI.add_kwargs(**task_settings)
        @TI.register_fl_task(model='my_model', data_loader='train_loader',
                device='device', optimizer='my_Adam_opt')
        def foo(my_model, train_loader, my_Adam_opt, device, batch_size, some_arg=356):
            # training or validation logic
        ...

FL tasks return a dictionary object with metrics: :code:`{metric name: metric value for this task}`.

.. note::
    The |productName| interactive API currently allows registering only standalone functions defined in the main module or imported from other modules inside the workspace.
    
    The :code:`TaskInterface` class must be instantiated before you can use its methods to register FL tasks.
    
        - :code:`@TI.register_fl_task()` needs tasks argument names for :code:`model`, :code:`data_loader`, :code:`device` , and :code:`optimizer` (optional) that constitute a *task contract*. This method adds the callable and the task contract to the task registry.
        - :code:`@TI.add_kwargs()` should be used to set up arguments that are not included in the contract.


.. _experiment_api_datainterface:

Register Federated Data Loader ( :code:`DataInterface` )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A *shard descriptor* defines how to read and format the local data. Therefore, the *data loader* contains the batching and augmenting data logic, which are common for all collaborators.

Subclass :code:`DataInterface` and implement the following methods.

    .. code-block:: python

        class CustomDataLoader(DataInterface):
            def __init__(self, **kwargs):
                # Initialize superclass with kwargs: this array will be passed
                # to get_data_loader methods
                super().__init__(**kwargs)
                # Set up augmentation, save required parameters,
                # use it as you regular dataset class
                validation_fraction = kwargs.get('validation_fraction', 0.5)
                ...
                
            @property
            def shard_descriptor(self):
                return self._shard_descriptor
                
            @shard_descriptor.setter
            def shard_descriptor(self, shard_descriptor):
                self._shard_descriptor = shard_descriptor
                # You can implement data splitting logic here
                # Or update your data set according to local Shard Descriptor atributes if required

            def get_train_loader(self, **kwargs):
                # these are the same kwargs you provided to __init__,
                # But passed on a collaborator machine
                bs = kwargs.get('train_batch_size', 32)
                return foo_loader()

            # so on, see the full list of methods below


The following are shard descriptor setter and getter methods:

    - :code:`shard_descriptor(self, shard_descriptor)` is called during the *Collaborator* initialization procedure with the local shard descriptor. Include in this method any logic that is triggered with the shard descriptor replacement.
    - :code:`get_train_loader(self, **kwargs)` is called before the execution of training tasks. This method returns the outcome of the training task according to the :code:`data_loader` contract argument. The :code:`kwargs` dict returns the same information that was provided during the :code:`DataInterface` initialization.
    - :code:`get_valid_loader(self, **kwargs)` is called before the execution of validation tasks. This method returns the outcome of the validation task according to the :code:`data_loader` contract argument. The :code:`kwargs` dict returns the same information that was provided during the :code:`DataInterface` initialization. 
    - :code:`get_train_data_size(self)` returns the number of samples in the local dataset for training. Use the information provided by the shard descriptor to determine how to split your training and validation tasks.
    - :code:`get_valid_data_size(self)` returns the number of samples in the local dataset for validation.
    

.. note::
    
    - The *User Dataset* class should be instantiated to pass further to the *Experiment* object. 
    - Dummy *shard descriptor* (or a custom local one) may be set up to test the augmentation or batching pipeline. 
    - Keyword arguments used during initialization on the frontend node may be used during dataloaders construction on collaborator machines.



.. _federation_api_start_fl_experiment:

Start an FL Experiment
======================

Use the Experiment API to prepare a workspace archive to transfer to the *Director*. 

    .. code-block:: python

        FLExperiment.start()

  .. note::
    Instances of interface classes :code:`(TaskInterface, DataInterface, ModelInterface)` must be passed to :code:`FLExperiment.start()` method along with other parameters. 
    
    This method:

        - Compiles all provided settings to a Plan object. The Plan is the central place where all actors in federation look up their parameters.
        - Saves **plan.yaml** to the :code:`plan` folder inside the workspace.
        - Serializes interface objects on the disk.
        - Prepares **requirements.txt** for remote Python environment setup.
        - Compresses the whole workspace to an archive.
        - Sends the experiment archive to the *Director* so it may distribute the archive across the federation and start the *Aggregator*.

FLExperiment :code:`start()` Method Parameters
----------------------------------------------

The following are parameters of the :code:`start()` method in FLExperiment:

:code:`model_provider`
    This parameter is defined earlier by the :code:`ModelInterface` object.

:code:`task_keeper`
    This parameter is defined earlier by the :code:`TaskInterface` object.

:code:`data_loader`
    This parameter is defined earlier by the :code:`DataInterface` object.

:code:`rounds_to_train`
    This parameter defines the number of aggregation rounds needed to be conducted before the experiment is considered finished.
    
:code:`delta_updates`
    This parameter sets up the aggregation to use calculated gradients instead of model checkpoints.

:code:`opt_treatment` 
    This parameter defines the optimizer state treatment in the federation. The following are available values:
    
    - **RESET**: the optimizer state is initialized each round from noise
    - **CONTINUE_LOCAL**: the optimizer state will be reused locally by every collaborator
    - **CONTINUE_GLOBAL**: the optimizer's state will be aggregated
    
:code:`device_assignment_policy`
    The following are available values:
    
    - **CPU_ONLY**: the :code:`device` parameter (which is a part of a task contract) that is passed to an FL task each round will be **cpu**
    - **CUDA_PREFFERED**: the :code:`device` parameter will be **cuda:{index}** if CUDA devices are enabled in the Envoy config and **cpu** otherwise.


.. _federation_api_observe_fl_experiment:

Observe the Experiment Execution
================================

If the experiment was accepted by the *Director*, you can oversee its execution with the :code:`FLexperiment.stream_metrics()` method. This method prints metrics from the FL tasks (and saves TensorBoard logs).


.. _federation_api_complete_fl_experiment:

Complete the Experiment
=======================

When the experiment has completed:

    - retrieve trained models in the native format using :code:`FLexperiment.get_best_model()` and :code:`FLexperiment.get_last_model()`.
    - erase experiment artifacts from the Director with :code:`FLexperiment.remove_experiment_data()`.
    
    
You may use the same federation object to report another experiment or even schedule several experiments that will be executed in series.
