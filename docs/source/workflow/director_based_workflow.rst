.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _director_workflow:

************************
Director-based workflow
************************

.. toctree::
    :maxdepth: 2
 
    establishing_federation_director_
    interactive_api_


.. _establishing_federation_director:

Establishing a long-living Federation with Director
#######################################

1. Install |productName| 
==================

Make sure that you installed |productName| in your virtual Python environment.
If not, use the instruction :ref:`install_initial_steps`.

2. Implement Shard Descriptors
==================

Then the data owners need to implement `Shard Descriptors` Python classes. 

|productName| framework provides a ‘Shard descriptor’ interface that should be described on every collaborator node 
to provide a unified data interface for FL experiments. Abstract “Shard descriptor” should be subclassed and 
all its methods should be implemented to describe the way data samples and labels will be loaded from disk 
during training. Shard descriptor is a subscriptable object that implements :code:`__getitem__()` and :code:`len()` methods 
as well as several additional methods to access ‘sample shape’, ‘target shape’, and ‘shard description’ text 
that may be used to identify participants during experiment definition and execution.

3. (Optional) Create certificates using Step-CA 
==================

The use of mTLS is strongly recommended for deployments in untrusted environments to establish participant identity and 
to encrypt communication. Users may either import certificates provided by their organization or utilize 
:ref:`PKI <semi_automatic_certification>` provided by |productName|.

4. Start Director
==================

Director is a central component in the Federation. It should be started on a node with at least two open ports. 
Learn more about the Director component here: :ref:`openfl_ll_components`

Create Director workspace
-------------------

Director requires a folder to operate in. Recieved experiments will be deployed in this folder. 
Moreover, supplementary files like Director's config files and certificates may be stored in this folder. 
One may use CLI command to create a structured workspace for Director with a default config file.

    .. code-block:: console

        $ fx director create-workspace -p director_ws

Tune Director config
-------------------

Director should be started from a config file. Basic config file should contain the Director's node FQDN, an open port, 
and :code:`sample_shape` and :code:`target_shape` fields with string representation of the unified data interface in the Federation. 
But it also may contain paths to certificates. 

Use CLI to start Director
-------------------

When the Director's config has been set up, one may use CLI to start the Director. Without mTLS protection:

    .. code-block:: console

       $ fx director start --disable-tls -c director_config.yaml

In the case of a certified Federation:

    .. code-block:: console

       $ fx director start -c director_config.yaml \
            -rc cert/root_ca.crt \
            -pk cert/priv.key \
            -oc cert/open.crt

5. Start Envoys
==================

Envoys are |productName|'s 'agents' on collaborator nodes that may recieve an experiment archive and provide 
access to local data.
When started Envoy will try to connect to the Director.

Create Envoy workspace
-------------------

The Envoy component also requires a folder to operate in. Use the following CLI command to create a workspace 
with convenient folder structure and default Envoy's config and Shard Descriptor Python script:

    .. code-block:: console

        $ fx envoy create-workspace -p envoy_ws

Setup Envoy's config
-------------------

Unlike Director’s config, the one for Envoy should contain settings for the local Shard Descriptor. 
The template field must be filled with the address of the local Shard Descriptor class, and settings filed 
should list arbitrary settings required to initialize the Shard Descriptor.

Use CLI to start Envoy
-------------------

To start the Envoy without mTLS use the following CLI command: 

    .. code-block:: console

        $ fx envoy start -n env_one --disable-tls \
            --shard-config-path shard_config.yaml -d director_fqdn:port

Alternatively, use the following command to establish a secured connection:

    .. code-block:: console

        $ ENVOY_NAME=envoy_example_name

        $ fx envoy start -n "$ENVOY_NAME" \
            --shard-config-path shard_config.yaml \
            -d director_fqdn:port -rc cert/root_ca.crt \
            -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt


6. Describing an FL experimnet using Interactive Python API
====================================

At this point, data scientists may register their experiments to be executed in the federation. 
|productName| provides a separate frontend Director’s client and :ref:`Interactive Python API <interactive_api>` 
to register experiments. 


.. _interactive_api:

Beta: |productName| Interactive Python API
#######################################

The |productName| Python Interactive API should help data scientists to adapt single node training code for 
running in the FL manner. The process of defining an FL experimnent is fully decoupled from the establishing 
a Federation routine. Everything that a data scientist needs to prepare an experiment is a Python interpreter and access to the Director.   

Python Interactive API Concepts
===============================

Workspace
----------
To initialize the workspace, create an empty folder and a Jupyter notebook (or a Python script) inside it. Root folder of the notebook will be considered as the workspace.
If some objects are imported in the notebook from local modules, source code should be kept inside the workspace.
If one decides to keep local test data inside the workspace, :code:`data` folder should be used as it will not be exported.
If one decides to keep certificates inside the workspace, :code:`cert` folder should be used as it will not be exported.
Only relevant source code or resources should be kept inside the workspace, since it will be zipped and transferred to collaborator machines.

Python Environment
---------------------
Create a virtual Python environment. Please, install only packages that are required for conducting the experiment, since Python environment will be replicated on collaborator nodes.



Defining a Federated Learning Experiment
========================================

Interactive API allows to register and start an FL experiment from a single entry point - a Jupyter notebook or a Python script.
An FL experiment definition process includes setting up several interface entities and experiment parameters.

Federation API
----------------
*Federation* entity is introduced to register and keep information about collaborators settings and their local data, 
as well as network settings to enable communication inside the federation. 
Each federation is bound to some Machine Learning problem in a sense that all collaborators dataset shards should 
follow the same annotation format for all samples. Once you created a federation, it may be used in several 
subsequent experiments.

To set up a federation, use Federation Interactive API.

.. code-block:: python

    from openfl.interface.interactive_api.federation import Federation

Federation API class should be initialized with the aggregator node FQDN and encryption settings. User may disable mTLS in trusted environments or provide paths to the certificate chain of CA, aggregator certificate and private key to enable mTLS.

.. code-block:: python

    federation = Federation(
        client_id: str, director_node_fqdn: str, director_port: str
        tls: bool, ca_cert_chain: str, cert: str, private_key: str)

* Federation's :code:`get_dummy_shard_descriptor` method should be used to create a fummy Shard Descriptor that 
  fakes access to real data. It may be used for debugging the user's experiment pipeline.
* Federation's :code:`get_shard_registry` method returns information about the envoys connected to the Director 
  and their Shard Descriptors.

Experiment API
----------------

*Experiment* entity allows registering training related objects, FL tasks and settings.
To set up an FL experiment someone should use the Experiment interactive API. 

.. code-block:: python

    from openfl.interface.interactive_api.experiment import FLExperiment

*Experiment* is being initialized by taking a Federation object and the experiment name as parameters.

.. code-block:: python

    fl_experiment = FLExperiment(federation: Federation, experiment_name: str)

To start an experiment user must register *DataLoader*, *Federated Learning tasks* and *Model* with *Optimizer*. 
There are several supplementary interface classes for these entities.

.. code-block:: python

    from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface

Registering model and optimizer
--------------------------------

First, user instantiate and initilize a model and optimizer in their favorite Deep Learning framework. 
Please, note that for now interactive API supports only *Keras* and *PyTorch* off-the-shelf.
Initialized model and optimizer objects then should be passed to the :code:`ModelInterface` along with the 
path to correct Framework Adapter plugin inside |productName| package. If desired DL framework is not covered by 
existing plugins, user can implement the plugin's interface and point :code:`framework_plugin` to the implementation 
inside the workspace.

.. code-block:: python

    from openfl.interface.interactive_api.experiment import ModelInterface
    MI = ModelInterface(model, optimizer, framework_plugin: str)

Registering FL tasks
---------------------

|productName| has a specific concept of an FL task.
Interactive API currently allows registering only standalone functions defined in the main module or 
imported from other modules inside the workspace.
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

:code:`DataInterface` is provided to support seamless remote data adaption.

As the *Shard Descriptor's* responsibilities are reading and formating the local data, the *DataLoader* is expected to 
contain batching and augmenting data logic, common for all collaborators. 

User must subclass :code:`DataInterface` and implement the following methods:

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

* Shard Descriptor setter and getter methods:
  :code:`shard_descriptor(self, shard_descriptor)` setter is the most important method. It will be called during the *Collaborator* 
  initialization procedure with the local Shard Descriptor. Any logic that is triggered with the Shard Descriptor replacement 
  must be also put here.
* :code:`get_train_loader(self, **kwargs)` will be called before training tasks execution. This method must return anything the user expects to receive in the training task with :code:`data_loader` contract argument. :code:`kwargs` dict holds the same information that was provided during :code:`DataInterface` initialization.
* :code:`get_valid_loader(self, **kwargs)` - see the point above (just replace training with validation)
* :code:`get_train_data_size(self)` - return number of samples in local train dataset. Use the information provided by Shard Descriptor, take into account your train / validation split. 
* :code:`get_valid_data_size(self)` - return number of samples in local validation dataset. 

User Dataset class should be instantiated to pass further to the *Experiment* object. Dummy *Shard Descriptor* 
(or a custom local one) may be set up to test the augmentation or batching pipeline.
Keyword arguments used during initialization on the frontend node may be used during dataloaders construction on collaborator machines.


Starting an FL experiment
========================================
Now we may use :code:`Experiment` API to prepare a workspace archive for transferring to the *Director*. In order to run *Collaborators*, we want to replicate the workspace and the Python environment 
on remote machines.

Instances of interface classes :code:`(TaskInterface, DataInterface, ModelInterface)` must be passed to :code:`FLExperiment.start()` method along with other parameters. 

This method:

* Compiles all provided settings to a Plan object. The Plan is the central place where all actors in federation look up their parameters.
* Saves plan.yaml to the :code:`plan/` folder inside the workspace.
* Serializes interface objects on the disk.
* Prepares :code:`requirements.txt` for remote Python environment setup.
* Compresses the whole workspace to an archive.
* Sends the experiment archive to the Director so it may distribute the archive across the Federation and start the *Aggregator*.
  
Observing the Experiment execution
----------------------------------

If the Experiment was accepted by the *Director* user can oversee its execution with 
:code:`FLexperiment.stream_metrics()` method that will is able to print metrics from the FL tasks (and save tensorboard logs).

When the Experiment is finished, user may retrieve trained models in the native format using :code:`FLexperiment.get_best_model()` 
and :code:`FLexperiment.get_last_model()` metods.

:code:`FLexperiment.remove_experiment_data()` allows erasing the experiment's artifacts from the Director.

When the Experiment is finished
----------------------------------

Users may utilize the same Federation object to report another experiment or even schedule several experiments that 
will be executed one by one.