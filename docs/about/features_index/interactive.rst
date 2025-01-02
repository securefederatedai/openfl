.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_interactive:

============================
Interactive API (Deprecated)
============================

A director-based workflow uses long-lived components in a federation. These components continue to be available to distribute more experiments in the federation.

- The *Director* is the central node of the federation. This component starts an *Aggregator* for each experiment, sends data to connected collaborator nodes, and provides updates on the status.
- The *Envoy* runs on collaborator nodes connected to the *Director*. When the *Director* starts an experiment, the *Envoy* starts the *Collaborator* to train the global model.


The director-based workflow comprises the following roles and their tasks:

    - `Director Manager: Set Up the Director`_
    - `Collaborator Manager: Set Up the Envoy`_
    - `Experiment Manager: Describe an Experiment`_

Follow the procedure in the director-based workflow to become familiar with the setup required and APIs provided for each role in the federation: *Experiment manager (Data scientist)*, *Director manager*, and *Collaborator manager*.

- *Experiment manager* (or Data scientist) is a person or group of people using OpenFL.
- *Director Manager* is ML model creator's representative controlling Director.
- *Collaborator manager* is Data owner's representative controlling Envoy.

.. note::
    The Open Federated Learning (OpenFL) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\  notebook or a Python\*\  script.

    See `Interactive Python API (Beta)`_ for details.

An overview of this workflow is shown below.

.. figure:: ../../source/openfl/director_workflow.svg

.. centered:: Overview of the Director-Based Workflow


.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0


.. _establishing_federation_director:

Director Manager: Set Up the Director
-------------------------------------

The *Director manager* sets up the *Director*, which is the central node of the federation.

    - :ref:`plan_agreement_director`
    - :ref:`optional_step_create_pki_using_step_ca`
    - :ref:`step0_install_director_prerequisites`
    - :ref:`step1_start_the_director`

.. _plan_agreement_director:

OPTIONAL STEP: Director's Plan Agreement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to carry out a secure federation, the Director must approve the FL Plan before starting the experiment. This check could be enforced with the use of the setting :code:`review_experiment: True` in director config. Refer to **director_config_review_exp.yaml** file under **PyTorch_Histology** interactive API example.
After the Director approves the experiment, it starts the aggregator and sends the experiment archive to all the participanting Envoys for review.
On the other hand, if the Director rejects the experiment, the experiment is aborted right away, no aggregator is started and the Envoys don't receive the experiment archive at all.

.. _optional_step_create_pki_using_step_ca:

OPTIONAL STEP: Create PKI Certificates Using Step-CA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of mutual Transport Layer Security (mTLS) is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or generate certificates with the :ref:`semi-automatic PKI <semi_automatic_certification>` provided by OpenFL.

.. _step0_install_director_prerequisites:

STEP 1: Install Open Federated Learning (OpenFL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install OpenFL in a virtual Python\*\  environment. See :ref:`installation` for details.

.. _step1_start_the_director:

STEP 2: Start the Director
^^^^^^^^^^^^^^^^^^^^^^^^^^

Start the Director on a node with at least two open ports. See :ref:`openfl_ll_components` to learn more about the Director entity.

1. Create a Director workspace with a default config file.

    .. code-block:: shell

        $ fx director create-workspace -p path/to/director_workspace_dir

 This workspace will contain received experiments and supplementary files (Director config file and certificates).

2. Modify the Director config file according to your federation setup.

 The default config file contains the Director node FQDN, an open port, path of certificates, and :code:`sample_shape` and :code:`target_shape` fields with string representation of the unified data interface in the federation.

3. Start the Director.

 If mTLS protection is not set up, run this command.

    .. code-block:: shell

       $ fx director start --disable-tls -c director_config.yaml

 If you have a federation with PKI certificates, run this command.

    .. code-block:: shell

       $ fx director start -c director_config.yaml \
            -rc cert/root_ca.crt \
            -pk cert/priv.key \
            -oc cert/open.crt



.. _establishing_federation_envoy:

Collaborator Manager: Set Up the Envoy
--------------------------------------

The *Collaborator manager* sets up the *Envoys*, which are long-lived components on collaborator nodes. When started, Envoys will try to connect to the Director. Envoys receive an experiment archive and provide access to local data.
    
    - :ref:`plan_agreement_envoy`
    - :ref:`optional_step_sign_pki_envoy`
    - :ref:`step0_install_envoy_prerequisites`
    - :ref:`step1_start_the_envoy`

.. _plan_agreement_envoy:

OPTIONAL STEP: Envoy's Plan Agreement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to carry out a secure federation, each of the Envoys must approve the experiment before it is started, after the Director's approval. This check could be enforced with the use of the parameter :code:`review_experiment: True` in envoy config. Refer to **envoy_config_review_exp.yaml** file under **PyTorch_Histology** interactive API example.
If any of the Envoys rejects the experiment, a :code:`set_experiment_failed` request is sent to the Director to stop the aggregator.

.. _optional_step_sign_pki_envoy:

OPTIONAL STEP: Sign PKI Certificates (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of mTLS is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or use the :ref:`semi-automatic PKI certificate <semi_automatic_certification>` provided by OpenFL.


.. _step0_install_envoy_prerequisites:

STEP 1: Install OpenFL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install OpenFL in a Python\*\ virtual environment. See :ref:`installation` for details.


.. _step1_start_the_envoy:

STEP 2: Start the Envoy
^^^^^^^^^^^^^^^^^^^^^^^

1. Create an Envoy workspace with a default config file and shard descriptor Python\*\  script.

    .. code-block:: shell

        $ fx envoy create-workspace -p path/to/envoy_workspace_dir

2. Modify the Envoy config file and local shard descriptor template.

    - Provide the settings field with the arbitrary settings required to initialize the shard descriptor.
    - Complete the shard descriptor template field with the address of the local shard descriptor class.

    .. note::
        The shard descriptor is an object to provide a unified data interface for FL experiments.
        The shard descriptor implements :code:`get_dataset()` method as well as several additional
        methods to access **sample shape**, **target shape**, and **shard description** that may be used to identify
        participants during experiment definition and execution.

        :code:`get_dataset()` method accepts the dataset_type (for instance train, validation, query, gallery) and returns
        an iterable object with samples and targets.

        User's implementation of ShardDescriptor should be inherented from :code:`openfl.interface.interactive_api.shard_descriptor.ShardDescriptor`. It should implement :code:`get_dataset`, :code:`sample_shape` and :code:`target_shape` methods to describe the way data samples and labels will be loaded from disk during training.

3. Start the Envoy.

 If mTLS protection is not set up, run this command.

    .. code-block:: shell

        ENVOY_NAME=envoy_example_name

        $ fx envoy start \
            -n "$ENVOY_NAME" \
            --disable-tls \
            --envoy-config-path envoy_config.yaml \
            -dh director_fqdn \
            -dp port

 If you have a federation with PKI certificates, run this command.

    .. code-block:: shell

        $ ENVOY_NAME=envoy_example_name

        $ fx envoy start \
            -n "$ENVOY_NAME" \
            --envoy-config-path envoy_config.yaml \
            -dh director_fqdn \
            -dp port \
            -rc cert/root_ca.crt \
            -pk cert/"$ENVOY_NAME".key \
            -oc cert/"$ENVOY_NAME".crt


.. _establishing_federation_experiment_manager:

Experiment Manager: Describe an Experiment
------------------------------------------

The process of defining an experiment is decoupled from the process of establishing a federation.
The Experiment manager (or data scientist) is able to prepare an experiment in a Python environment.
Then the Experiment manager registers experiments into the federation using `Interactive Python API (Beta)`_
that is allow to communicate with the Director using a gRPC client.


.. _interactive_python_api:

Interactive Python API (Beta)
-----------------------------

The Open Federated Learning (OpenFL) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\  notebook or a Python script.

    - `Prerequisites`_
    - `Define a Federated Learning Experiment`_
    - `Federation API`_
    - `Experiment API`_
    - `Start an FL Experiment`_


.. _prerequisites:

Prerequisites
^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The definition process of a federated learning experiment uses the interactive Python API to set up several interface entities and experiment parameters.

The following are the interactive Python API to define an experiment:

    - `Federation API`_
    - `Experiment API`_
    - `Start an FL Experiment`_
    - `Observe the Experiment Execution`_

.. note::
    Each federation is bound to some Machine Learning problem in a sense that all collaborators dataset shards should allow to solve the same data science problem.
    For example object detection and semantic segmentation problems should be solved in different federations. \


.. _federation_api:

Federation API
""""""""""""""

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
""""""""""""""

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

Instantiate and initialize a model and optimizer in your preferred deep learning framework.

    .. code-block:: python

        from openfl.interface.interactive_api.experiment import ModelInterface
        MI = ModelInterface(model, optimizer, framework_plugin: str)

The initialized model and optimizer objects should be passed to the :code:`ModelInterface` along with the path to correct Framework Adapter plugin inside the OpenFL package
or from local workspace.

.. note::
    The OpenFL interactive API supports *TensorFlow* and *PyTorch* frameworks via existing plugins.
    User can add support for other deep learning frameworks via the plugin interface and point to your implementation of a :code:`framework_plugin` in :code:`ModelInterface`.


.. _experiment_api_taskinterface:

Register FL Tasks ( :code:`TaskInterface` )

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
    The OpenFL interactive API currently allows registering only standalone functions defined in the main module or imported from other modules inside the workspace.

    The :code:`TaskInterface` class must be instantiated before you can use its methods to register FL tasks.

        - :code:`@TI.register_fl_task()` needs tasks argument names for :code:`model`, :code:`data_loader`, :code:`device` , and :code:`optimizer` (optional) that constitute a *task contract*. This method adds the callable and the task contract to the task registry.
        - :code:`@TI.add_kwargs()` should be used to set up arguments that are not included in the contract.


.. _experiment_api_datainterface:

Register Federated Data Loader ( :code:`DataInterface` )

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
^^^^^^^^^^^^^^^^^^^^^^

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
""""""""""""""""""""""""""""""""""""""""""""""

The following are parameters of the :code:`start()` method in FLExperiment:

:code:`model_provider`
    This parameter is defined earlier by the :code:`ModelInterface` object.

:code:`task_keeper`
    This parameter is defined earlier by the :code:`TaskInterface` object.

:code:`data_loader`
    This parameter is defined earlier by the :code:`DataInterface` object.

:code:`task_assigner`
    This parameter is optional. You can pass a `Custom task assigner function`_.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the experiment was accepted by the *Director*, you can oversee its execution with the :code:`FLexperiment.stream_metrics()` method. This method prints metrics from the FL tasks (and saves TensorBoard logs).

.. _federation_api_get_fl_experiment_status:

Get Experiment Status
^^^^^^^^^^^^^^^^^^^^^

You can get the current experiment status with the :code:`FLexperiment.get_experiment_status()` method. The status could be pending, in progress, finished, rejected or failed.

.. _federation_api_complete_fl_experiment:

Complete the Experiment
^^^^^^^^^^^^^^^^^^^^^^^

When the experiment has completed:

    - retrieve trained models in the native format using :code:`FLexperiment.get_best_model()` and :code:`FLexperiment.get_last_model()`.
    - erase experiment artifacts from the Director with :code:`FLexperiment.remove_experiment_data()`.


You may use the same federation object to report another experiment or even schedule several experiments that will be executed in series.

Custom task assigner function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OpenFL has an entity named Task Assigner, that responsible for aggregator task assigning to collaborators.
There are three default tasks that are used: :code:`train`, :code:`locally_tuned_model_validate`,
:code:`aggregated_model_validate`.
When you register a train function and pass optimizer it generates a train task:

    .. code-block:: python

        task_keeper = TaskInterface()


        @task_keeper.register_fl_task(model='net_model', data_loader='train_loader',
                                      device='device', optimizer='optimizer')
        def train(net_model, train_loader, optimizer, device, loss_fn=cross_entropy, some_parameter=None):
            torch.manual_seed(0)
            ...

When you register a validate function, it generates two tasks: :code:`locally_tuned_model_validate` and
:code:`aggregated_model_validate`.
:code:`locally_tuned_model_validate` is applied by collaborator to locally trained model,
:code:`aggregated_model_validate` - to a globally aggregated model.
If there not a train task only aggregated_model_validate are generated.

Since 1.3 version it is possible to create a custom task assigner function to implement your own task assigning logic.
You can get registered task from :code:`task_keeper` calling method :code:`get_registered_tasks`:

    .. code-block:: python

        tasks = task_keeper.get_registered_tasks()


And  then implement your own assigner function:

    .. code-block:: python

        def random_assigner(collaborators, round_number, **kwargs):
            """Assigning task groups randomly while ensuring target distribution"""
            import random
            random.shuffle(collaborators)
            collaborator_task_map = {}
            for idx, col in enumerate(collaborators):
                # select only 70% collaborators for training and validation, 30% for validation
                if (idx+1)/len(collaborators) <= 0.7:
                    collaborator_task_map[col] = tasks.values()  # all three tasks
                else:
                    collaborator_task_map[col] = [tasks['aggregated_model_validate']]
            return collaborator_task_map

And then pass that function to fl_experiment start method:
    .. code-block:: python

        fl_experiment.start(
            model_provider=model_interface,
            task_keeper=task_keeper,
            data_loader=fed_dataset,
            task_assigner=random_assigner,
            rounds_to_train=50,
            opt_treatment='CONTINUE_GLOBAL',
            device_assignment_policy='CUDA_PREFERRED'
        )


It will be passed to assigner and tasks will be assigned to collaborators by using this function.

Another example.
If you want only exclude some collaborators from experiment, you can define next assigner function:

    .. code-block:: python

        def filter_assigner(collaborators, round_number, **kwargs):
            collaborator_task_map = {}
            exclude_collaborators = ['env_two', 'env_three']
            for collaborator_name in collaborators:
                if collaborator_name in exclude_collaborators:
                    continue
                collaborator_task_map[collaborator_name] = [
                    tasks['train'],
                    tasks['locally_tuned_model_validate'],
                    tasks['aggregated_model_validate']
                ]
            return collaborator_task_map


Also you can use static shard information to exclude any collaborators without cuda devices from training:

    .. code-block:: python

        shard_registry = federation.get_shard_registry()
        def filter_by_shard_registry_assigner(collaborators, round_number, **kwargs):
            collaborator_task_map = {}
            for collaborator in collaborators:
                col_status = shard_registry.get(collaborator)
                if not col_status or not col_status['is_online']:
                    continue
                node_info = col_status['shard_info'].node_info
                # Assign train task if collaborator has GPU with total memory more that 8 GB
                if len(node_info.cuda_devices) > 0 and node_info.cuda_devices[0].memory_total > 8 * 1024**3:
                    collaborator_task_map[collaborator] = [
                        tasks['train'],
                        tasks['locally_tuned_model_validate'],
                        tasks['aggregated_model_validate'],
                    ]
                else:
                    collaborator_task_map[collaborator] = [
                        tasks['aggregated_model_validate'],
                    ]
            return collaborator_task_map


Assigner with additional validation round:

    .. code-block:: python

        rounds_to_train = 3
        total_rounds = rounds_to_train + 1 # use fl_experiment.start(..., rounds_to_train=total_rounds,...)

        def assigner_with_last_round_validation(collaborators, round_number, **kwargs):
            collaborator_task_map = {}
            for collaborator in collaborators:
                if round_number == total_rounds - 1:
                    collaborator_task_map[collaborator] = [
                        tasks['aggregated_model_validate'],
                    ]
                else:
                    collaborator_task_map[collaborator] = [
                        tasks['train'],
                        tasks['locally_tuned_model_validate'],
                        tasks['aggregated_model_validate']
                    ]
            return collaborator_task_map


.. toctree
..    overview.how_can_intel_protect_federated_learning
..    overview.what_is_intel_federated_learning
