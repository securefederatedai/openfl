.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation:

******************
Run the Federation
******************

The following are workflows you can consider when creating a federated learning setup.

:doc:`source/workflow/director_based_workflow`
    A federation created with this workflow continues to be available to distribute more experiments in series.

:doc:`source/workflow/running_the_federation.agg_based`
    With this workflow, the federation is terminated when the experiment is finished.

.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _director_workflow:

************************
Director-Based Workflow
************************

A director-based workflow uses long-lived components in a federation. These components continue to be available to distribute more experiments in the federation.

- The *Director* is the central node of the federation. This component starts an *Aggregator* for each experiment, sends data to connected collaborator nodes, and provides updates on the status.
- The *Envoy* runs on collaborator nodes connected to the *Director*. When the *Director* starts an experiment, the *Envoy* starts the *Collaborator* to train the global model.


The director-based workflow comprises the following roles and their tasks:

    - :ref:`establishing_federation_director`
    - :ref:`establishing_federation_envoy`
    - :ref:`establishing_federation_experiment_manager`

Follow the procedure in the director-based workflow to become familiar with the setup required and APIs provided for each role in the federation: *Director manager*, *Collaborator manager*, and *Experiment manager (data scientist)*.

.. note::
    The Open Federated Learning (|productName|) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\  notebook or a Python\*\  script.

    See :doc:`director_based_workflow.interactive_api` for details.

An overview of this workflow is shown below.

.. figure:: ./source/openfl/static_diagram.svg

.. centered:: Overview of the Director-Based Workflow


.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0


.. _establishing_federation_director:

Director Manager: Set Up the Director
=====================================

The *Director manager* sets up the *Director*, which is the central node of the federation.

    - :ref:`optional_step_create_pki_using_step_ca`
    - :ref:`step0_install_director_prerequisites`
    - :ref:`step1_start_the_director`

.. _optional_step_create_pki_using_step_ca:

OPTIONAL STEP: Create PKI Certificates Using Step-CA (Optional)
--------------------------------------------------------

The use of mutual Transport Layer Security (mTLS) is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or generate certificates with the :ref:`semi-automatic PKI <semi_automatic_certification>` provided by |productName|.

.. _step0_install_director_prerequisites:

STEP 0: Install Open Federated Learning (|productName|)
-------------------------------------------------------

Install |productName| in a virtual Python\*\  environment. See :ref:`install_package` for details.

.. _step1_start_the_director:

STEP 1: Start the Director
--------------------------

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
======================================

The *Collaborator manager* sets up the *Envoys*, which are long-lived components on collaborator nodes. When started, Envoys will try to connect to the Director. Envoys receive an experiment archive and provide access to local data.

    - :ref:`optional_step_sign_pki_envoy`
    - :ref:`step0_install_envoy_prerequisites`
    - :ref:`step1_start_the_envoy`

.. _optional_step_sign_pki_envoy:

OPTIONAL STEP: Sign PKI Certificates (Optional)
--------------------------------------------------------

The use of mTLS is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or use the :ref:`semi-automatic PKI certificate <semi_automatic_certification>` provided by |productName|.


.. _step0_install_envoy_prerequisites:

STEP 0: Install |productName|
-----------------------------

Install |productName| in a Python\*\ virtual environment. See :ref:`install_package` for details.


.. _step1_start_the_envoy:

STEP 3: Start the Envoy
-----------------------

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

        MyShardDescriptor should be inherent from :code:`openfl.interface.interactive_api.shard_descriptor.ShardDescriptor`. It should implements :code:`get_dataset`, :code:`sample_shape` and :code:`target_shape` methods o describe the way data samples and labels will be loaded from disk during training.

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
==========================================

The process of defining an experiment is decoupled from the process of establishing a federation.
The Experiment manager (or data scientist) is able to prepare an experiment in a Python environment.
Then the Experiment manager registers experiments into the federation using :ref:`Interactive Python API <interactive_api>`
that is allow to communicate with Director using a gRPC client.

.. _running_the_federation_aggregator_based:

*************************
Aggregator-Based Workflow
*************************

An overview of this workflow is shown below.

.. figure:: /images/openfl_flow.png

.. centered:: Overview of the Aggregator-Based Workflow

There are two ways to run federation without Director:

- :ref:`Bare metal approach <running_the_federation_manual>`
- :ref:`Docker approach <running_the_federation_docker>`


This workflow uses short-lived components in a federation, which is terminated when the experiment is finished. The components are as follows:

- The *Collaborator* uses a local dataset to train a global model and the *Aggregator* receives model updates from *Collaborators* and aggregate them to create the new global model.
- The *Aggregator* is framework-agnostic, while the *Collaborator* can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_\* \  or `PyTorch <https://pytorch.org/>`_\*\.


For this workflow, you modify the federation workspace to your requirements by editing the Federated Learning plan (FL plan) along with the Python\*\  code that defines the model and the data loader. The FL plan is a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file that defines the collaborators, aggregator, connections, models, data, and any other parameters that describe the training.


.. _plan_settings:

******************************************
Federated Learning Plan (FL Plan) Settings
******************************************

.. note::
    Use the Federated Learning plan (FL plan) to modify the federation workspace to your requirements in an **aggregator-based workflow**.


The FL plan is described by the **plan.yaml** file located in the **plan** directory of the workspace.


Each YAML top-level section contains the following subsections:

- ``template``: The name of the class including top-level packages names. An instance of this class is created when plan gets initialized.
- ``settings``: The arguments that are passed to the class constructor.
- ``defaults``: The file that contains default settings for this subsection.
  Any setting from defaults file can be overriden in the **plan.yaml** file.

The following is an example of a **plan.yaml**:

.. literalinclude:: ../openfl-workspace/torch_cnn_mnist/plan/plan.yaml
  :language: yaml


Configurable Settings
=====================

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
-----

Each task subsection contains the following:

- ``function``: The function name to call.
  The function must be the one defined in :class:`TaskRunner <openfl.federated.TaskRunner>` class.
- ``kwargs``: kwargs passed to the ``function``.

.. note::
    See an `example <https://github.com/intel/openfl/blob/develop/openfl/federated/task/runner.py>`_ of the :class:`TaskRunner <openfl.federated.TaskRunner>` class for details.


.. _running_the_federation_manual:

*******************
Bare Metal Approach
*******************

.. note::

    Ensure you have installed the |productName| package on every node (aggregator and collaborators) in the federation.

    See :ref:`install_package` for details.


You can use the `"Hello Federation" bash script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.sh>`_ to quickly create a federation (an aggregator node and two collaborator nodes) to test the project pipeline.

.. literalinclude:: ../tests/github/test_hello_federation.sh
  :language: bash

However, continue with the following procedure for details in creating a federation with an aggregator-based workflow.

    :doc:`STEP 1: Create a Workspace on the Aggregator <running_the_federation.baremetal>`

        - Creates a federated learning workspace on one of the nodes.


    :doc:`STEP 2: Configure the Federation <running_the_federation.certificates>`

        - Ensures each node in the federation has a valid public key infrastructure (PKI) certificate.
        - Distributes the workspace from the aggregator node to the other collaborator nodes.


    :doc:`STEP 3: Start the Federation <running_the_federation.start_nodes>`


.. _creating_workspaces:

********************************************
STEP 1: Create a Workspace on the Aggregator
********************************************

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


    This command initializes the FL plan and autopopulates the `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ of the aggregator node. This FQDN is embedded within the FL plan so the collaborator nodes know the address of the externally accessible aggregator server to connect to.

    If you have connection issues with the autopopulated FQDN in the FL plan, you can do **one of the following**:

	- OPTION 1: override the autopopulated FQDN value with the :code:`-a` flag.

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



.. _instruction_manual_certs:

********************************
STEP 2: Configure the Federation
********************************

The objectives in this step:

    - Ensure each node in the federation has a valid public key infrastructure (PKI) certificate. See :doc:`/source/utilities/pki` for details on available workflows.
    - Distribute the workspace from the aggregator node to the other collaborator nodes.


.. _install_certs_agg:

On the Aggregator Node
======================

Setting Up the Certificate Authority
------------------------------------

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
-----------------------

1. Export the workspace so that it can be imported to the collaborator nodes.

    .. code-block:: console

       fx workspace export

   The :code:`export` command will archive the current workspace (with a :code:`zip` file extension) and create a **requirements.txt** of the current Python\*\ packages in the virtual environment.

2. The next step is to transfer this workspace archive to each collaborator node.


.. _install_certs_colab:

On the Collaborator Nodes
=========================

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

****************************
STEP 3: Start the Federation
****************************

On the Aggregator Node
======================

1. Start the Aggregator.

    .. code-block:: console

       fx aggregator start

 Now, the Aggregator is running and waiting for Collaborators to connect.

.. _running_collaborators:

On the Collaborator Nodes
=========================

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

********************
Docker\* \  Approach
********************

There are two ways you can run |productName| with Docker\*\.

- :ref:`Deploy a federation in a Docker container <running_the_federation_docker_base_image>`
- :ref:`Deploy the workspace in a Docker container <running_the_federation_docker_workspace>`


.. _running_the_federation_docker_base_image:

Option 1: Deploy a Federation in a Docker Container
===================================================

Prerequisites
-------------

You have already built an |productName| image. See :doc:`../../install.docker` for details.

Procedure
---------

1. Run the |productName| image.

    .. code-block:: console

       docker run -it --network host openfl


You can now experiment with |productName| in the container. For example, you can test the project pipeline with the `"Hello Federation" bash script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.sh>`_.




.. _running_the_federation_docker_workspace:

Option 2: Deploy Your Workspace in a Docker Container
=====================================================

Prerequisites
-------------

You have already set up a TaskRunner and run :code:`fx plan initialize` in the workspace directory. See :ref:`Create a Workspace on the Aggregator <creating_workspaces>` for details.

Procedure
---------

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

4. :doc:`Start the federation <running_the_federation.start_nodes>`.



.. toctree::
   :maxdepth: 2
   :hidden:

   source/workflow/running_the_federation.agg_based
   source/workflow/director_based_workflow

