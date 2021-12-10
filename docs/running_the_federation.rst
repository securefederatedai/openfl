.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation:

******************
Run the Federation
******************

OpenFL currently supports two types of workflow for how to set up and run a federation: Director-based workflow (preferrable) and Aggregator-based workflow (old workflow, will not be supported soon). Director-based workflow introduces a new and more convenient way to set up a federation and brings "long-lived" components in a federation ("Director" and "Envoy").

`Director-Based Workflow`_
    A federation created with this workflow continues to be available to distribute more experiments in series.

`Aggregator-Based Workflow`_
    With this workflow, the federation is terminated when the experiment is finished.


.. _director_workflow:


Director-Based Workflow
=======================

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
- *Collaborator manager* is Data onwer's representative controlling Envoy.

.. note::
    The Open Federated Learning (|productName|) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\  notebook or a Python\*\  script.

    See `Interactive Python API (Beta)`_ for details.

An overview of this workflow is shown below.

.. figure:: ./source/openfl/director_workflow.svg

.. centered:: Overview of the Director-Based Workflow


.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0


.. _establishing_federation_director:

Director Manager: Set Up the Director
-------------------------------------

The *Director manager* sets up the *Director*, which is the central node of the federation.

.. _optional_step_create_pki_using_step_ca:

OPTIONAL STEP: Create PKI Certificates Using Step-CA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of mutual Transport Layer Security (mTLS) is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or generate certificates with the :ref:`semi-automatic PKI <semi_automatic_certification>` provided by |productName|.

.. _step0_install_director_prerequisites:

STEP 1: Install Open Federated Learning (|productName|)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install |productName| in a virtual Python\*\  environment. See :ref:`install_package` for details.

.. _step1_start_the_director:

STEP 2: Start the Director
^^^^^^^^^^^^^^^^^^^^^^^^^^

Start the Director on a node with at least two open ports. See :ref:`openfl_ll_components` to learn more about the Director entity.

1. Create a Director workspace with a default config file.

    .. code-block:: console

        fx director create-workspace -p path/to/director_workspace_dir

 This workspace will contain received experiments and supplementary files (Director config file and certificates).

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

Collaborator Manager: Set Up the Envoy
--------------------------------------

The *Collaborator manager* sets up the *Envoys*, which are long-lived components on collaborator nodes. When started, Envoys will try to connect to the Director. Envoys receive an experiment archive and provide access to local data.

    - :ref:`optional_step_sign_pki_envoy`
    - :ref:`step0_install_envoy_prerequisites`
    - :ref:`step1_start_the_envoy`

.. _optional_step_sign_pki_envoy:

OPTIONAL STEP: Sign PKI Certificates (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of mTLS is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or use the :ref:`semi-automatic PKI certificate <semi_automatic_certification>` provided by |productName|.


.. _step0_install_envoy_prerequisites:

STEP 1: Install |productName|
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install |productName| in a Python\*\ virtual environment. See :ref:`install_package` for details.


.. _step1_start_the_envoy:

STEP 2: Start the Envoy
^^^^^^^^^^^^^^^^^^^^^^^

1. Create an Envoy workspace with a default config file and shard descriptor Python\*\  script.

    .. code-block:: console

        fx envoy create-workspace -p path/to/envoy_workspace_dir

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

    .. code-block:: console

        ENVOY_NAME=envoy_example_name

        fx envoy start \
            -n "$ENVOY_NAME" \
            --disable-tls \
            --envoy-config-path envoy_config.yaml \
            -dh director_fqdn \
            -dp port

 If you have a federation with PKI certificates, run this command.

    .. code-block:: console

        ENVOY_NAME=envoy_example_name

        fx envoy start \
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

The Open Federated Learning (|productName|) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\  notebook or a Python script.

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

The initialized model and optimizer objects should be passed to the :code:`ModelInterface` along with the path to correct Framework Adapter plugin inside the |productName| package
or from local workspace.

.. note::
    The |productName| interactive API supports *TensorFlow* and *PyTorch* frameworks via existing plugins.
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
    The |productName| interactive API currently allows registering only standalone functions defined in the main module or imported from other modules inside the workspace.

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


.. _federation_api_complete_fl_experiment:

Complete the Experiment
^^^^^^^^^^^^^^^^^^^^^^^

When the experiment has completed:

    - retrieve trained models in the native format using :code:`FLexperiment.get_best_model()` and :code:`FLexperiment.get_last_model()`.
    - erase experiment artifacts from the Director with :code:`FLexperiment.remove_experiment_data()`.


You may use the same federation object to report another experiment or even schedule several experiments that will be executed in series.



.. _running_the_federation_aggregator_based:

Aggregator-Based Workflow
=========================

An overview of this workflow is shown below.

.. figure:: /images/openfl_flow.png

.. centered:: Overview of the Aggregator-Based Workflow

There are two ways to run federation without Director:

- `Bare Metal Approach`_
- `Docker Approach`_


This workflow uses short-lived components in a federation, which is terminated when the experiment is finished. The components are as follows:

- The *Collaborator* uses a local dataset to train a global model and the *Aggregator* receives model updates from *Collaborators* and aggregates them to create the new global model.
- The *Aggregator* is framework-agnostic, while the *Collaborator* can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_\* \  or `PyTorch <https://pytorch.org/>`_\*\.


For this workflow, you modify the federation workspace to your requirements by editing the Federated Learning plan (FL plan) along with the Python\*\  code that defines the model and the data loader. The FL plan is a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file that defines the collaborators, aggregator, connections, models, data, and any other parameters that describe the training.


.. _plan_settings:


Federated Learning Plan (FL Plan) Settings
------------------------------------------

.. note::
    Use the Federated Learning plan (FL plan) to modify the federation workspace to your requirements in an **aggregator-based workflow**.


The FL plan is described by the **plan.yaml** file located in the **plan** directory of the workspace.


Each YAML top-level section contains the following subsections:

- ``template``: The name of the class including top-level packages names. An instance of this class is created when the plan gets initialized.
- ``settings``: The arguments that are passed to the class constructor.
- ``defaults``: The file that contains default settings for this subsection.
  Any setting from defaults file can be overridden in the **plan.yaml** file.

The following is an example of a **plan.yaml**:

.. literalinclude:: ../openfl-workspace/torch_cnn_mnist/plan/plan.yaml
  :language: yaml


Configurable Settings
^^^^^^^^^^^^^^^^^^^^^

- :class:`Aggregator <openfl.component.Aggregator>`
    `openfl.component.Aggregator <https://github.com/intel/openfl/blob/develop/openfl/component/aggregator/aggregator.py>`_

- :class:`Collaborator <openfl.component.Collaborator>`
    `openfl.component.Collaborator <https://github.com/intel/openfl/blob/develop/openfl/component/collaborator/collaborator.py>`_

- :class:`Data Loader <openfl.federated.data.loader.DataLoader>`
    `openfl.federated.data.loader.DataLoader <https://github.com/intel/openfl/blob/develop/openfl/federated/data/loader.py>`_

- :class:`Task Runner <openfl.federated.task.runner.TaskRunner>`
    `openfl.federated.task.runner.TaskRunner <https://github.com/intel/openfl/blob/develop/openfl/federated/task/runner.py>`_

- :class:`Assigner <openfl.component.Assigner>`
    `openfl.component.Assigner <https://github.com/intel/openfl/blob/develop/openfl/component/assigner/assigner.py>`_


Tasks
^^^^^

Each task subsection contains the following:

- ``function``: The function name to call.
  The function must be the one defined in :class:`TaskRunner <openfl.federated.TaskRunner>` class.
- ``kwargs``: kwargs passed to the ``function``.

.. note::
    See an `example <https://github.com/intel/openfl/blob/develop/openfl/federated/task/runner.py>`_ of the :class:`TaskRunner <openfl.federated.TaskRunner>` class for details.


.. _running_the_federation_manual:


.. _interactive_api:



Bare Metal Approach
-------------------

.. note::

    Ensure you have installed the |productName| package on every node (aggregator and collaborators) in the federation.

    See :ref:`install_package` for details.


You can use the `"Hello Federation" bash script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.sh>`_ to quickly create a federation (an aggregator node and two collaborator nodes) to test the project pipeline.

.. literalinclude:: ../tests/github/test_hello_federation.sh
  :language: bash

However, continue with the following procedure for details in creating a federation with an aggregator-based workflow.

    `STEP 1: Create a Workspace on the Aggregator`_

        - Creates a federated learning workspace on one of the nodes.


    `STEP 2: Configure the Federation`_

        - Ensures each node in the federation has a valid public key infrastructure (PKI) certificate.
        - Distributes the workspace from the aggregator node to the other collaborator nodes.


    `STEP 3: Start the Federation`_


.. _creating_workspaces:


STEP 1: Create a Workspace on the Aggregator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.	Start a Python 3.8 (>=3.6, <3.9) virtual environment and confirm |productName| is available.

	.. code-block:: python

		fx


2. 	This example uses the :code:`keras_cnn_mnist` template.

	Set the environment variables to use the :code:`keras_cnn_mnist` as the template and :code:`${HOME}/my_federation` as the path to the workspace directory.

    .. code-block:: console

        export WORKSPACE_TEMPLATE=keras_cnn_mnist
        export WORKSPACE_PATH=${HOME}/my_federation

3.	Decide a workspace template, which are end-to-end federated learning training demonstrations. The following is a sample of available templates:

 - :code:`keras_cnn_mnist`: a workspace with a simple `Keras <http://keras.io/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`tf_2dunet`: a workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will use the `BraTS <https://www.med.upenn.edu/sbia/brats2017/data.html>`_ dataset and train in a federation.
 - :code:`tf_cnn_histology`: a workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_histology`: a workspace with a simple `PyTorch <http://pytorch.org/>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_mnist`: a workspace with a simple `PyTorch <http://pytorch.org>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.

  See the complete list of available templates.

    .. code-block:: console

       fx workspace create --prefix ${WORKSPACE_PATH}


4.  Create a workspace directory for the new federation project.

    .. code-block:: console

       fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}


    .. note::

		You can use your own models by overwriting the Python scripts in the **src** subdirectory in the workspace directory.

5.  Change to the workspace directory.

    .. code-block:: console

        cd ${WORKSPACE_PATH}

6.  Install the workspace requirements:

    .. code-block:: console

        pip install -r requirements.txt


7.	Create an initial set of random model weights.

    .. note::

        While models can be trained from scratch, in many cases the federation performs fine-tuning of a previously trained model. For this reason, pre-trained weights for the model are stored in protobuf files on the aggregator node and passed to collaborator nodes during initialization.

        The protobuf file with the initial weights is found in **${WORKSPACE_TEMPLATE}_init.pbuf**.


    .. code-block:: console

		fx plan initialize


    This command initializes the FL plan and auto populates the `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ of the aggregator node. This FQDN is embedded within the FL plan so the collaborator nodes know the address of the externally accessible aggregator server to connect to.

    If you have connection issues with the auto populated FQDN in the FL plan, you can do **one of the following**:

	- OPTION 1: override the auto populated FQDN value with the :code:`-a` flag.

		.. code-block:: console

			fx plan initialize -a aggregator-hostname.internal-domain.com

	- OPTION 2: override the apparent FQDN of the system by setting an FQDN environment variable.

		.. code-block:: console

			export FQDN=x.x.x.x

		and initializing the FL plan

		.. code-block:: console

			fx plan initialize


.. note::

       Each workspace may have multiple FL plans and multiple collaborator lists associated with it. Therefore, :code:`fx plan initialize` has the following optional parameters.

       +-------------------------+---------------------------------------------------------+
       | Optional Parameters     | Description                                             |
       +=========================+=========================================================+
       | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
       +-------------------------+---------------------------------------------------------+
       | -c, --cols_config PATH  | Authorized collaborator list [default = plan/cols.yaml] |
       +-------------------------+---------------------------------------------------------+
       | -d, --data_config PATH  | The data set/shard configuration file                   |
       +-------------------------+---------------------------------------------------------+



.. _configure_the_federation:


STEP 2: Configure the Federation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The objectives in this step:

    - Ensure each node in the federation has a valid public key infrastructure (PKI) certificate. See :doc:`/source/utilities/pki` for details on available workflows.
    - Distribute the workspace from the aggregator node to the other collaborator nodes.


.. _install_certs_agg:

On the Aggregator Node
""""""""""""""""""""""

Setting Up the Certificate Authority

1. Change to the path of your workspace:

    .. code-block:: console

       cd WORKSPACE_PATH

2. Set up the aggregator node as the `certificate authority <https://en.wikipedia.org/wiki/Certificate_authority>`_ for the federation.

 All certificates will be signed by the aggregator node. Follow the instructions and enter the information as prompted. The command will create a simple database file to keep track of all issued certificates.

    .. code-block:: console

       fx workspace certify

3. Run the aggregator certificate creation command, replacing :code:`AFQDN` with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator node.

    .. code-block:: console

       fx aggregator generate-cert-request --fqdn AFQDN

    .. note::

       On Linux\*\, you can discover the FQDN with this command:

           .. code-block:: console

              hostname --all-fqdns | awk '{print $1}'

   .. note::

      You can override the apparent FQDN of the system by setting an FQDN environment variable before creating the certificate.

        .. code-block:: console

            fx aggregator generate-cert-request export FQDN=x.x.x.x

      If you omit the :code:`--fdqn` parameter, then :code:`fx` will automatically use the FQDN of the current node assuming the node has been correctly set with a static address.

        .. code-block:: console

            fx aggregator generate-cert-request

4. Run the aggregator certificate signing command, replacing :code:`AFQDN` with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator node.

    .. code-block:: console

       fx aggregator certify --fqdn AFQDN


   .. note::

      You can override the apparent FQDN of the system by setting an FQDN environment variable (:code:`export FQDN=x.x.x.x`) before signing the certificate.

        .. code-block:: console

           fx aggregator certify export FQDN=x.x.x.x

5. This node now has a signed security certificate as the aggregator for this new federation. You should have the following files.

    +---------------------------+--------------------------------------------------+
    | File Type                 | Filename                                         |
    +===========================+==================================================+
    | Certificate chain         | WORKSPACE.PATH/cert/cert_chain.crt               |
    +---------------------------+--------------------------------------------------+
    | Aggregator certificate    | WORKSPACE.PATH/cert/server/agg_{AFQDN}.crt         |
    +---------------------------+--------------------------------------------------+
    | Aggregator key            | WORKSPACE.PATH/cert/server/agg_{AFQDN}.key         |
    +---------------------------+--------------------------------------------------+

    where **AFQDN** is the fully-qualified domain name of the aggregator node.

.. _workspace_export:

Exporting the Workspace


1. Export the workspace so that it can be imported to the collaborator nodes.

    .. code-block:: console

       fx workspace export

   The :code:`export` command will archive the current workspace (with a :code:`zip` file extension) and create a **requirements.txt** of the current Python\*\ packages in the virtual environment.

2. The next step is to transfer this workspace archive to each collaborator node.


.. _install_certs_colab:

On the Collaborator Node
""""""""""""""""""""""""

Importing the Workspace

1. Copy the :ref:`workspace archive <workspace_export>` from the aggregator node to the collaborator nodes.

2. Import the workspace archive.

    .. code-block:: console

       fx workspace import --archive WORKSPACE.zip

 where **WORKSPACE.zip** is the name of the workspace archive. This will unzip the workspace to the current directory and install the required Python packages within the current virtual environment.

3. For each test machine you want to run as collaborator nodes, create a collaborator certificate request to be signed by the certificate authority.

 Replace :code:`COL_LABEL` with the label you assigned to the collaborator. This label does not have to be the FQDN; it can be any unique alphanumeric label.

    .. code-block:: console

       fx collaborator generate-cert-request -n {COL_LABEL}


 The creation script will also ask you to specify the path to the data. For this example, enter the integer that represents which MNIST shard to use on this collaborator node. For the first collaborator node enter **1**. For the second collaborator node enter **2**.

 This will create the following files:

    +-----------------------------+--------------------------------------------------------+
    | File Type                   | Filename                                               |
    +=============================+========================================================+
    | Collaborator CSR            | WORKSPACE.PATH/cert/client/col_{COL_LABEL}.csr         |
    +-----------------------------+--------------------------------------------------------+
    | Collaborator key            | WORKSPACE.PATH/cert/client/col_{COL_LABEL}.key         |
    +-----------------------------+--------------------------------------------------------+
    | Collaborator CSR Package    | WORKSPACE.PATH/col_{COL_LABEL}_to_agg_cert_request.zip |
    +-----------------------------+--------------------------------------------------------+


4. On the aggregator node (i.e., the certificate authority in this example), sign the Collaborator CSR Package from the collaborator nodes.

    .. code-block:: console

       fx collaborator certify --request-pkg /PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip

   where :code:`/PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip` is the path to the Collaborator CSR Package containing the :code:`.csr` file from the collaborator node. The certificate authority will sign this certificate for use in the federation.

   The command packages the signed collaborator certificate, along with the **cert_chain.crt** file needed to verify certificate signatures, for transport back to the collaborator node:

    +---------------------------------+------------------------------------------------------------+
    | File Type                       | Filename                                                   |
    +=================================+============================================================+
    | Certificate and Chain Package   | WORKSPACE.PATH/agg_to_col_{COL_LABEL}_signed_cert.zip      |
    +---------------------------------+------------------------------------------------------------+

5. On the collaborator node, import the signed certificate and certificate chain into your workspace.

    .. code-block:: console

       fx collaborator certify --import /PATH/TO/agg_to_col_{COL_LABEL}_signed_cert.zip



.. _running_the_federation.start_nodes:


STEP 3: Start the Federation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the Aggregator Node
""""""""""""""""""""""

1. Start the Aggregator.

    .. code-block:: console

       fx aggregator start

 Now, the Aggregator is running and waiting for Collaborators to connect.

.. _running_collaborators:

On the Collaborator Nodes
"""""""""""""""""""""""""

1. Open a new terminal, change the directory to the workspace, and activate the virtual environment.

2. Run the Collaborator.

    .. code-block:: console

       fx collaborator start -n {COLLABORATOR_LABEL}

    where :code:`COLLABORATOR_LABEL` is the label for this Collaborator.

    .. note::

       Each workspace may have multiple FL plans and multiple collaborator lists associated with it.
       Therefore, :code:`fx collaborator start` has the following optional parameters.

           +-------------------------+---------------------------------------------------------+
           | Optional Parameters     | Description                                             |
           +=========================+=========================================================+
           | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
           +-------------------------+---------------------------------------------------------+
           | -d, --data_config PATH  | The data set/shard configuration file                   |
           +-------------------------+---------------------------------------------------------+

3. Repeat the earlier steps for each collaborator node in the federation.

  When all of the Collaborators connect, the Aggregator starts training. You will see log messages describing the progress of the federated training.

  When the last round of training is completed, the Aggregator stores the final weights in the protobuf file that was specified in the YAML file, which in this example is located at **save/${WORKSPACE_TEMPLATE}_latest.pbuf**.



.. _running_the_federation_docker:


Docker Approach
---------------

There are two ways you can run |productName| with Docker\*\.

- `Option 1: Deploy a Federation in a Docker Container`_
- `Option 2: Deploy Your Workspace in a Docker Container`_


.. _running_the_federation_docker_base_image:

Option 1: Deploy a Federation in a Docker Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prerequisites
"""""""""""""

You have already built an |productName| image. See :ref:`install_docker` for details.

Procedure
"""""""""""""

1. Run the |productName| image.

    .. code-block:: console

       docker run -it --network host openfl


You can now experiment with |productName| in the container. For example, you can test the project pipeline with the `"Hello Federation" bash script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.sh>`_.


.. _running_the_federation_docker_workspace:

Option 2: Deploy Your Workspace in a Docker Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prerequisites
"""""""""""""

You have already set up a TaskRunner and run :code:`fx plan initialize` in the workspace directory. See `STEP 1: Create a Workspace on the Aggregator`_ for details.

Procedure
"""""""""

1. Build an image with the workspace you created.

    .. code-block:: console

       fx workspace dockerize


    By default, the image is saved as **WORKSPACE_NAME_image.tar** in the workspace directory.

2. The image can be distributed and run on other nodes without any environment preparation.

    .. parsed-literal::

        docker run -it --rm \\
            --network host \\
            -v user_data_folder:/home/user/workspace/data \\
            ${WORKSPACE_IMAGE_NAME} \\
            bash


    .. note::

        The FL plan should be initialized with the FQDN of the node where the aggregator container will be running.

3. Generate public key infrastructure (PKI) certificates for all collaborators and the aggregator. See :doc:`/source/utilities/pki` for details.

4. `STEP 3: Start the Federation`_.

