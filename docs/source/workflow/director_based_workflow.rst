.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _director_workflow:

************************
Director-Based Workflow
************************

.. toctree::
    :maxdepth: 2
 
    overview_director_
	establishing_federation_director_
    interactive_api_

.. _overview_director:

Overview
========

The director-based workflow uses long-lived components in a federation. These components continue to be available to distribute more experiments in the federation.
	
- The *Director* is the central node of the federation. This component starts an *aggregator* for each experiment, sends data to connected collaborator nodes, and provides updates on the status.
- The *Envoy* runs on collaborator nodes connected to the *Director*. When the *Director* starts an experiment, the *Envoy* starts the *collaborator* to train the global model.

.. note::

	Follow the procedure in the director-based workflow to become familiar with the setup required and APIs provided for each role in the federation: *Director manager*, *Envoy manager*, and *Experiment manager (data scientist)*. 


.. figure:: ../openfl/static_diagram.svg

.. centered:: Overview of the Director-Based Workflow

.. _establishing_federation_director:

Director Manager: Set Up the Director
=====================================

The *Director manager* sets up the *Director*, which is the central node of the federation.

    - :ref:`step1_install_director_prerequisites`
    - :ref:`STEP 2: Create Public Key Infrastructure (PKI) Certificate Using Step-CA (Optional) <step2_create_pki_using_step_ca>`
    - :ref:`step3_start_the_director`


.. _step1_install_director_prerequisites:

STEP 1: Install |productName| 
-----------------------------

Install |productName| in a virtual Python\*\ environment. See :ref:`install_package` for details.

.. _step2_create_pki_using_step_ca:

STEP 2: Create PKI Certificates Using Step-CA (Optional)
--------------------------------------------------------

The use of mTLS is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or use the :ref:`semi-automatic PKI certificate <semi_automatic_certification>` provided by |productName|.

.. _step3_start_the_director:

STEP 3: Start the Director
--------------------------

Director is a central component in the Federation. It should be started on a node with at least two open ports. See :ref:`openfl_ll_components` to learn more.

1. Create a Director workspace with a default config file.

    .. code-block:: console

        fx director create-workspace -p director_ws
        
 This workspace contains received experiments and supplementary files (Director config file and certificates).

2. Modify the Director config file according to your federation setup.

 The default config file contains the Director node FQDN, an open port, path of certificates, and :code:`sample_shape` and :code:`target_shape` fields with string representation of the unified data interface in the federation.
 
3. Start the Director.

 If mTLS protection is not set up, run this command.
 
    .. code-block:: console

       fx director start --disable-tls -c director_config.yaml
 
 If you have a federation with PKI certificates, run this command.
 
    .. code-block:: console

       fx director start -c director_config.yaml \
            -rc cert/root_ca.crt \
            -pk cert/priv.key \
            -oc cert/open.crt



.. _establishing_federation_envoy:

Envoy Manager: Set Up the Envoy
===============================

The *Envoy manager* sets up the *Envoys*, which are long-lived components on collaborator nodes. Envoys receive an experiment archive and provide access to local data. When started, Envoys will try to connect to the Director.

    - :ref:`step1_install_envoy_prerequisites`
    - :ref:`STEP 2: Sign Public Key Infrastructure (PKI) Certificate (Optional) <step2_sign_pki_envoy>`
    - :ref:`step3_start_the_envoy`

.. _step1_install_envoy_prerequisites:

STEP 1: Install |productName| 
-----------------------------

Install |productName| in a virtual Python\*\ environment. See :ref:`install_package` for details.

.. _step2_sign_pki_envoy:

STEP 2: Sign PKI Certificates (Optional)
--------------------------------------------------------

The use of mTLS is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or use the :ref:`semi-automatic PKI certificate <semi_automatic_certification>` provided by |productName|.


.. _step3_start_the_envoy:

STEP 3: Start the Envoy
-----------------------

1. Create an Envoy workspace with a default config file and shard descriptor Python\*\ script.

    .. code-block:: console

        fx envoy create-workspace -p envoy_ws

2. Modify the config file and local shard descriptor template.

    - Provide the settings field with the arbitrary settings required to initialize the shard descriptor.
    - Complete the shard descriptor template field with the address of the local shard descriptor class.

    .. note::
        The shard descriptor is a scriptable object to provide a unified data interface for FL experiments. The shard descriptor implements :code:`__getitem__()` and :code:`len()` methods as well as several additional methods to access **sample shape**, **target shape**, and **shard description** that may be used to identify participants during experiment definition and execution.
        
        Abstract shard descriptor should be subclassed and all its methods should be implemented to describe the way data samples and labels will be loaded from disk during training. 
        
3. Start the Envoy.

 If mTLS protection is not set up, run this command.
 
    .. code-block:: console

        fx envoy start -n env_one --disable-tls \
            --shard-config-path shard_config.yaml -d director_fqdn:port

 If you have a federation with PKI certificates, run this command.
 
    .. code-block:: console

        ENVOY_NAME=envoy_example_name

        fx envoy start -n "$ENVOY_NAME" \
            --shard-config-path shard_config.yaml \
            -d director_fqdn:port -rc cert/root_ca.crt \
            -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt
            


.. _establishing_federation_experiment_manager:

Experiment Manager: Describe an Experiment
==========================================

The Experiment manager (or data scientist) registers experiments into the federation in the following manner:

    - frontend Director client
    - :ref:`Interactive Python API <interactive_api>`

The process of defining an experiment is decoupled from the process of establishing a federation. Everything that a data scientist needs to prepare an experiment is a Python interpreter and access to the Director.

.. _interactive_api:

*******************************************
|productName| Interactive Python API (Beta)
*******************************************

The |productName| interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\ notebook or a Python script. 

    - :ref:`federation_api_prerequisites`
    - :ref:`federation_api_define_fl_experiment`
    - :ref:`federation_api_start_fl_experiment`
    - :ref:`federation_api_observe_fl_experiment`
    - :ref:`federation_api_complete_fl_experiment`

.. _federation_api_prerequisites:

Prerequisites
=============

The Experiment manager requires the following:

Access to the Director.
    Initialize a workspace by creating an empty directory and placing inside the workspace a Jupyter\*\ notebook or a Python script.
    
    Items in the workspace may include:
    
        - source code of objects imported into the notebook from local modules
        - local test data stored in a **data** directory
        - certificates stored in a **cert** directory
        
    .. note::
    
        This workspace will be archived and transferred to collaborator nodes. Ensure only relevant source code or resources are stored in the workspace.


Python Intepreter
    Create a virtual Python environment with packages required for conducting the experiment. The Python environment is replicated on collaborator nodes.


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
    Each federation is bound to some Machine Learning problem in a sense that all collaborators dataset shards should follow the same annotation format for all samples. \



.. _federation_api:

Federation API
--------------

The *Federation* entity registers and keeps the following information:

    - Collaborator settings and local data
    - Network settings

1. Set up a federation.

    .. code-block:: python

        from openfl.interface.interactive_api.federation import Federation


    .. note::
        Once a federation is created, you may use the federation for subsequent experiments.


2. Initialize the API class with the aggregator node FQDN and encryption settings.

    .. code-block:: python

        federation = Federation(
            client_id: str, director_node_fqdn: str, director_port: str
            tls: bool, ca_cert_chain: str, cert: str, private_key: str)

    .. note::
        You may disable mTLS in trusted environments or enable mTLS by providing paths to the certificate chain of the certificate authority, aggregator certificate, and private key.


.. note::
    Methods available in the Federation API:
        
        - :code:`get_dummy_shard_descriptor`: creates a dummy shard descriptor for debugging the  the experiment pipeline
        - :code:`get_shard_registry`: returns information about the Envoys connected to the Director and their shard descriptors

.. _experiment_api:

Experiment API
----------------

The *Experiment* entity registers training-related objects, FL tasks, and settings.

1. Set up a federated learning experiment.

    .. code-block:: python

        from openfl.interface.interactive_api.experiment import FLExperiment

2. Initialize the experiment with the following parameters: a federation object and an experiment name.

    .. code-block:: python

        fl_experiment = FLExperiment(federation: Federation, experiment_name: str)

3. Register these supplementary interface classes: :code:`TaskInterface`, :code:`DataInterface`, and :code:`ModelInterface`.

    .. code-block:: python

        from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface


.. _experiment_api_modelinterface:

Register the Model and Optimizer ( :code:`ModelInterface` )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instantiate and initialize a model and optimizer in your preferred deep learning framework.

    .. code-block:: python

        from openfl.interface.interactive_api.experiment import ModelInterface
        MI = ModelInterface(model, optimizer, framework_plugin: str)
    
The initialized model and optimizer objects should be passed to the :code:`ModelInterface` along with the path to correct Framework Adapter plugin inside |productName| package.

.. note::
    The |productName| interactive API supports *Keras* and *PyTorch* models via existing plugins. You can implement other deep learning models via the plugin interface and point the :code:`framework_plugin` to your implementation. 


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
        def foo(my_model, train_loader, my_Adam_opt, device, batch_size, some_arg=356)
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

1. Use the Experiment API to prepare a workspace archive to transfer to the *Director*. 

    .. code-block:: python

        FLExperiment.start()

  .. note::
    Instances of interface classes :code:`(TaskInterface, DataInterface, ModelInterface)` must be passed to :code:`FLExperiment.start()` method along with other parameters. 
    
    This method:

        - Compiles all provided settings to a Plan object. The Plan is the central place where all actors in federation look up their parameters.
        - Saves **plan.yaml** to the :code:`plan/` folder inside the workspace.
        - Serializes interface objects on the disk.
        - Prepares **requirements.txt** for remote Python environment setup.
        - Compresses the whole workspace to an archive.
        - Sends the experiment archive to the *Director* so it may distribute the archive across the federation and start the *Aggregator*.


2. Replicate the workspace and Python environment on remote machines which will serve as *Collaborators*.


.. _federation_api_observe_fl_experiment:

Observe the Experiment Execution
================================

If the experiment was accepted by the *Director*, you can oversee its execution with the :code:`FLexperiment.stream_metrics()` method. This method prints metrics from the FL tasks (and saved tensorboard logs).


.. _federation_api_complete_fl_experiment:

Complete the Experiment
=======================

When the experiment has completed:

    - retrieve trained models in the native format using :code:`FLexperiment.get_best_model()` and :code:`FLexperiment.get_last_model()`.
    - erase experiment artifacts from the Director with :code:`FLexperiment.remove_experiment_data()`.
    
    
You may use the same federation object to report another experiment or even schedule several experiments that will be executed in series.