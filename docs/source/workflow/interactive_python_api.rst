.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _interactive_api:

#########################################################
Experimental: |productName| Interactive Python API
#########################################################

*********************************
Python Interactive API Concepts
*********************************

Workspace
==========
To initialize the workspace, create an empty folder and a Jupyter notebook (or a Python script) inside it. Root folder of the notebook will be considered as the workspace.
If some objects are imported in the notebook from local modules, source code should be kept inside the workspace.
If one decides to keep local test data inside the workspace, :code:`data` folder should be used as it will not be exported.
If one decides to keep certificates inside the workspace, :code:`cert` folder should be used as it will not be exported.
Only relevant source code or resources should be kept inside the workspace, since it will be zipped and transferred to collaborator machines.

Python Environment
===================
Create a virtual Python environment. Please, install only packages that are required for conducting the experiment, since Python environment will be replicated on collaborator nodes.

******************************************
Certification
******************************************
If you have trusted workspace and connection should not be encrypted you can use :code:`disable_tls` option while starting experiment.
Otherwise it is necessary to certify each node participating in the federation. Certificates allow to use mutual tls connection between nodes.
You can certify nodes by your own pki system or use pki provided by OpenFL. It is based on `step-ca <https://github.com/smallstep/certificates>`_
as a server and `step <https://github.com/smallstep/cli>`_ as a client utilities. They are downloaded from github during workspace setup. Regardless of the certification method,
paths to certificates on each node are provided at start of experiment. Pki workflow from OpenFL will be discussed below.

OpenFL PKI workflow
===================
Openfl PKI pipeline asumes creating local CA with https server which listen signing requests.
Certificates from each node can be signed by requesting to CA server with special token.
Token must be copied to each node by some secure way. Each step is considered in detail below.

1. Create CA, i.e create root key pair, CA server config and other.
    .. code-block:: console

       $ fx pki install -p </path/to/ca/dir> --password <123> --ca-url <host:port>
    | :code:`-p` - path to folder, which will contain ca files.
    | :code:`--password` - password that will encrypts some ca files.
    | :code:`--ca-url` - host and port which ca server will listen
    This command will also download `step-ca <https://github.com/smallstep/certificates>`_ and `step <https://github.com/smallstep/cli>`_ binaries from github.

2. Run CA https server.
    .. code-block:: console

       $ fx pki run -p </path/to/ca/dir>
    | :code:`-p` - path to folder, which will contain ca files.

3. Get token for some node.

    .. code-block:: console

       $ fx pki get-token -n <subject>
    | :code:`-n` - subject name, fqdn for director, collaborator name for envoy or api name for api-layer node

    Run this command on ca side, from ca folder. Output is a token which contains JWT (json web token) from CA server and CA
    root certificate concatenated together. This JWT have twenty-four hours time-to-live.

4. Copy token to node side (director or envoy) by some secure channel and run certify command.
    .. code-block:: console

       $ fx pki certify -n <subject> -t <token>
    | :code:`-n` - subject name, fqdn for director, collaborator name for envoy or api name for api-layer node
    | :code:`-t` - output token from previous command
    This command call step client, to connect to CA server over https.
    Https is provided by root certificate which was copy with JWT.
    Server authenticates client by JWT and client authenticates server by root certificate.

Now signed certificate and private key are stored on current node. Signed certificate has one year time-to-live. You should certify all node that will participate in federation: director, all envoys and api-layer node.

******************************************
Defining a Federated Learning Experiment
******************************************
Interactive API allows setting up an experiment from a single entrypoint - a Jupyter notebook or a Python script.
Defining an experiment includes setting up several interface entities and experiment parameters.

Federation API
===================
*Federation* entity is introduced to register and keep information about collaborators settings and their local data, as well as network settings to enable communication inside the federation. 
Each federation is bound to some Machine Learning problem in a sense that all collaborators dataset shards should follow the same annotation format for all samples. Once you created a federation, it may be used in several subsequent experiments.

To set up a federation, use Federation Interactive API.

.. code-block:: python

    from openfl.interface.interactive_api.federation import Federation

Federation API class should be initialized with the aggregator node FQDN and encryption settings. Someone may disable mTLS in trusted environments or provide paths to the certificate chain of CA, aggregator certificate and private key to enable mTLS.

.. code-block:: python

    federation = Federation(central_node_fqdn: str, tls: bool, cert_chain: str, agg_certificate: str, agg_private_key: str)

Federation's :code:`register_collaborators` method should be used to provide an information about collaborators participating in a federation.
It requires a dictionary object - :code:`{collaborator name : local data path}`.

Experiment API
===================

*Experiment* entity allows registering training related objects, FL tasks and settings.
To set up an FL experiment someone should use the Experiment interactive API. 

.. code-block:: python

    from openfl.interface.interactive_api.experiment import FLExperiment

*Experiment* is being initialized by taking federation as a parameter.

.. code-block:: python

    fl_experiment = FLExperiment(federation=federation)

To start an experiment user must register *DataLoader*, *Federated Learning tasks* and *Model* with *Optimizer*. There are several supplementary interface classes for these purposes.

.. code-block:: python

    from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface

Registering model and optimizer
--------------------------------

First, user instantiate and initilize a model and optimizer in their favorite Deep Learning framework. Please, note that for now interactive API supports only *Keras* and *PyTorch* off-the-shelf.
Initialized model and optimizer objects then should be passed to the :code:`ModelInterface` along with the path to correct Framework Adapter plugin inside OpenFL package. If desired DL framework is not covered by existing plugins, someone can implement the plugin's interface and point :code:`framework_plugin` to the implementation inside the workspace.

.. code-block:: python

    from openfl.interface.interactive_api.experiment import ModelInterface
    MI = ModelInterface(model=model_unet, optimizer=optimizer_adam, framework_plugin=framework_adapter)

Registering FL tasks
---------------------

We have an agreement on what we consider to be a FL task.
Interactive API currently allows registering only standalone functions defined in the main module or imported from other modules inside the workspace.
We also have requirements on task signature. Task should accept the following objects:

1. model - will be rebuilt with relevant weights for every task by `TaskRunner`
2. :code:`data_loader` - data loader that will provide local data
3. device - a device to be used for execution on collaborator machines
4. optimizer (optional) - model optimizer, only for training tasks

Moreover FL tasks should return a dictionary object with metrics :code:`{metric name: metric value for this task}`.

:code:`Task Interface` class is designed to register task and accompanying information.
This class must be instantiated, then it's special methods may be used to register tasks.

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


:code:`@TI.register_fl_task()` needs tasks argument names for (model, data_loader, device, optimizer (optional)) that constitute tasks 'contract'.
It adds the callable and the task contract to the task registry.

:code:`@TI.add_kwargs()` method should be used to set up those arguments that are not included in the contract.

Registering Federated DataLoader
---------------------------------

:code:`DataInterface` is provided to support a remote DataLoader initialization.

It is initialized with User Dataset class object and all the keyword arguments can be used by dataloaders during training or validation.
User must subclass :code:`DataInterface` and implements several methods.

* :code:`_delayed_init(self, data_path)` is the most important method. It will be called during collaborator initialization procedure with relevant :code:`data_path` (one that corresponds to the collaborator name that user registered in federation). User Dataset class should be instantiated with local :code:`data_path` here. If dataset initalization procedure differs for some of the  collaborators, the initialization logic must be described here. Dataset sharding procedure for test runs should also be described in this method. User is free to save objects in class fields for later use.
* :code:`get_train_loader(self, **kwargs)` will be called before training tasks execution. This method must return anything user expects to recieve in the training task with :code:`data_loader` contract argument. :code:`kwargs` dict holds the same information that was provided during :code:`DataInterface` initialization.
* :code:`get_valid_loader(self, **kwargs)` - see the point above only with validation data
* :code:`get_train_data_size(self)` - return number of samples in local train dataset.
* :code:`get_valid_data_size(self)` - return number of samples in local validation dataset. 

Preparing workspace distribution
---------------------------------
Now we may use :code:`Experiment` API to prepare a workspace archive for transferring to collaborator's node. In order to run a collaborator, we want to replicate the workspace and the Python environment.

Instances of interface classes :code:`(TaskInterface, DataInterface, ModelInterface)` must be passed to :code:`FLExperiment.prepare_workspace_distribution()` method along with other parameters. 

This method:

* Compiles all provided setings to a Plan object. This is the central place where all actors in federation look up their parameters.
* Saves plan.yaml to the :code:`plan/` folder inside the workspace.
* Serializes interface objects on the disk.
* Prepares :code:`requirements.txt` for remote Python environment setup.
* Compressess the workspace to an archive so it can be coppied to collaborator nodes.
  
Starting the aggregator
---------------------------

As all previous steps done, the experiment is ready to start
:code:`FLExperiment.start_experiment()` method requires :code:`model_interface` object with initialized weights.

It starts a local aggregator that will wait for collaborators to connect.

Starting collaborators
=======================

The process of starting collaborators has not changed.
User must transfer the workspace archive to a remote node and type in console:

.. code-block:: python

    fx workspace import --archive ws.zip

Please, note that aggregator and all the collaborator nodes should have the same Python interpreter version as the machine used for defining the experiment.

then cd to the workspace and run

.. code-block:: python

    fx collaborator start -d data.yaml -n one

For more details, please refer to the TaskRunner API section.