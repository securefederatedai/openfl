.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_task_runner:

================
TaskRunner API
================

This is a deep dive into the TaskRunner API. To gain familiarity with this API, we recommend going through the `quickstart <../../tutorials/taskrunner.html>`_ guide. Note that the quickstart guide is focused on simulating an experiment locally. The design choices of this API are best understood when transitioning from a local experiment to a distributed federation, which is how real-world federated learning experiments are conducted.

.. figure:: ../../images/openfl_flow.png

The Task Runner API uses short-lived components in a federation, which are terminated once the experiment finishes. These components are:

- The :code:`Collaborator` uses a local dataset to train a global model and the :code:`Aggregator` receives model updates from :code:`Collaborator` s and aggregates them to create the new global model.
- The :code:`Aggregator` is framework-agnostic, while the :code:`Collaborator` can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_\* \  or `PyTorch <https://pytorch.org/>`_\*\.


For this workflow, one needs modify the federation workspace to their requirements by editing the Federated Learning plan (FL plan) along with the Python\*\  code that defines the model and the data loader. The FL plan is a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file that defines the collaborators, aggregator, connections, models, data, and any other parameters that describe the training.


.. _plan_settings:


Federated Learning Plan (FL Plan) Settings
------------------------------------------

.. note::
    Use the Federated Learning plan (FL plan) to modify the federation workspace to your requirements in an **aggregator-based workflow**.


In order for participants to agree to take part in an experiment, everyone should know ahead of time both what code is going to run on their infrastructure and exactly what information on their system will be accessed. The federated learning (FL) plan aims to capture all of this information needed to decide whether to participate in an experiment, in addition to runtime details needed to load the code and make remote connections.
The FL plan is described by the **plan.yaml** file located in the **plan** directory of the workspace.

Configurable Settings
^^^^^^^^^^^^^^^^^^^^^

- :class:`Aggregator <openfl.component.Aggregator>`
    `openfl.component.Aggregator <https://github.com/intel/openfl/blob/develop/openfl/component/aggregator/aggregator.py>`_
    Defines the settings for the aggregator which is the model-owner in the experiment. While models can be trained from scratch, in many cases the federation performs fine-tuning of a previously trained model. For this reason, pre-trained weights for the model are stored in protobuf files on the aggregator node and passed to collaborator nodes during initialization. The settings for aggregator include:

 - :code:`init_state_path`: (str:path) Defines the weight protobuf file path where the experiment's initial weights will be loaded from. These weights will be generated with the `fx plan initialize` command.
 - :code:`best_state_path`: (str:path) Defines the weight protobuf file path that will be saved to for the highest accuracy model during the experiment.
 - :code:`last_state_path`: (str:path)  Defines the weight protobuf file path that will be saved to during the last round completed in each experiment.
 - :code:`rounds_to_train`: (int) Specifies the number of rounds in a federation. A federated learning round is defined as one complete iteration when the collaborators train the model and send the updated model weights back to the aggregator to form a new global model. Within a round, collaborators can train the model for multiple iterations called epochs.
 - :code:`write_logs`: (boolean) Metric logging callback feature. By default, logging is done through `tensorboard <https://www.tensorflow.org/tensorboard/get_started>`_ but users can also use custom metric logging function for each task.      


- :class:`Collaborator <openfl.component.Collaborator>`
    `openfl.component.Collaborator <https://github.com/intel/openfl/blob/develop/openfl/component/collaborator/collaborator.py>`_
    Defines the settings for the collaborator which is the data owner in the experiment. The settings for collaborator include:

 - :code:`delta_updates`: (boolean) Determines whether the difference in model weights between the current and previous round will be sent (True), or if whole checkpoints will be sent (False). Setting to delta_updates to True leads to higher sparsity in model weights sent across, which may improve compression ratios.
 - :code:`opt_treatment`: (str) Defines the optimizer state treatment policy. Valid options are : 'RESET' - reinitialize optimizer for every round (default), 'CONTINUE_LOCAL' - keep local optimizer state for every round, 'CONTINUE_GLOBAL' - aggregate optimizer state for every round.


- :class:`Data Loader <openfl.federated.data.loader.DataLoader>`
    `openfl.federated.data.loader.DataLoader <https://github.com/intel/openfl/blob/develop/openfl/federated/data/loader.py>`_
    Defines the data loader class that provides access to local dataset. It implements a train loader and a validation loader that takes in the train dataset and the validation dataset respectively. The settings for the dataloader include:

 - :code:`collaborator_count`: (int) The number of collaborators participating in the federation
 - :code:`data_group_name`: (str) The name of the dataset
 - :code:`batch_size`: (int) The size of the training or validation batch


- :class:`Task Runner <openfl.federated.task.runner.TaskRunner>`
    `openfl.federated.task.runner.TaskRunner <https://github.com/intel/openfl/blob/develop/openfl/federated/task/runner.py>`_
    Defines the model, training/validation functions, and how to extract and set the tensors from model weights and optimizer dictionary. Depending on different AI frameworks like PyTorch and Tensorflow, users can select pre-defined task runner methods.


- :class:`Assigner <openfl.component.Assigner>`
    `openfl.component.Assigner <https://github.com/intel/openfl/blob/develop/openfl/component/assigner/assigner.py>`_
    Defines the task that are sent to the collaborators from the aggregator. There are three default tasks that could be given to each Collaborator:
 
 - :code:`aggregated_model_validation`: (str) Perform validation on aggregated global model sent by the aggregator.
 - :code:`train`: (str) Perform training on the global model.
 - :code:`locally_tuned_model_validation`: (str) Perform validation on the model that was locally trained by the collaborator.


Each YAML top-level section contains the following subsections:

- ``template``: The name of the class including top-level packages names. An instance of this class is created when the plan gets initialized.
- ``settings``: The arguments that are passed to the class constructor.
- ``defaults``: The file that contains default settings for this subsection.
  Any setting from defaults file can be overridden in the **plan.yaml** file.

The following is an example of a **plan.yaml**:

.. literalinclude:: ../../../openfl-workspace/torch_cnn_mnist/plan/plan.yaml
  :language: yaml


Tasks
^^^^^

Each task subsection contains the following:

- ``function``: The function name to call.
  The function must be the one defined in :class:`TaskRunner <openfl.federated.TaskRunner>` class.
- ``kwargs``: kwargs passed to the ``function``.

.. note::
    See an `example <https://github.com/intel/openfl/blob/develop/openfl/federated/task/runner.py>`_ of the :class:`TaskRunner <openfl.federated.TaskRunner>` class for details.


.. _running_the_federation_manual:


.. _interactive_api (Deprecated):



Bare Metal Approach
-------------------

.. note::

    Ensure you have installed the OpenFL package on every node (aggregator and collaborators) in the federation.

    See :ref:`installation` for details.



    `STEP 1: Create a Workspace`_

        - Creates a federated learning workspace on one of the nodes.


    `STEP 2: Configure the Federation`_

        - Ensures each node in the federation has a valid public key infrastructure (PKI) certificate.
        - Distributes the workspace from the aggregator node to the other collaborator nodes.


    `STEP 3: Start the Federation`_


.. _creating_workspaces:


STEP 1: Create a Workspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.	Start a Python 3.10 (>=3.10, <3.13) virtual environment and confirm OpenFL is available.

	.. code-block:: shell

		$ fx


2. 	This example uses the :code:`keras_cnn_mnist` template.

	Set the environment variables to use the :code:`keras_cnn_mnist` as the template and :code:`${HOME}/my_federation` as the path to the workspace directory.

    .. code-block:: shell

        $ export WORKSPACE_TEMPLATE=keras_cnn_mnist
        $ export WORKSPACE_PATH=${HOME}/my_federation

3.	Decide a workspace template, which are end-to-end federated learning training demonstrations. The following is a sample of available templates:

 - :code:`keras_cnn_mnist`: a workspace with a simple `Keras <http://keras.io/>`__ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`tf_2dunet`: a workspace with a simple `TensorFlow <http://tensorflow.org>`__ CNN model that will use the `BraTS <https://www.med.upenn.edu/sbia/brats2017/data.html>`_ dataset and train in a federation.
 - :code:`tf_cnn_histology`: a workspace with a simple `TensorFlow <http://tensorflow.org>`__ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_histology`: a workspace with a simple `PyTorch <http://pytorch.org/>`__ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_mnist`: a workspace with a simple `PyTorch <http://pytorch.org>`__ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.

  See the complete list of available templates.

    .. code-block:: shell

       $ fx workspace create --prefix ${WORKSPACE_PATH}


4.  Create a workspace directory for the new federation project.

    .. code-block:: shell

       $ fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}


    .. note::

		You can use your own models by overwriting the Python scripts in the **src** subdirectory in the workspace directory.

5.  Change to the workspace directory.

    .. code-block:: shell

        $ cd ${WORKSPACE_PATH}

6.  Install the workspace requirements:

    .. code-block:: shell

        $ pip install -r requirements.txt


7.	Create an initial set of random model weights.

    .. note::

        While models can be trained from scratch, in many cases the federation performs fine-tuning of a previously trained model. For this reason, pre-trained weights for the model are stored in protobuf files on the aggregator node and passed to collaborator nodes during initialization.

        The protobuf file with the initial weights is found in **${WORKSPACE_TEMPLATE}_init.pbuf**.


    .. code-block:: shell

		$ fx plan initialize


    This command initializes the FL plan and auto populates the `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ of the aggregator node. This FQDN is embedded within the FL plan so the collaborator nodes know the address of the externally accessible aggregator server to connect to.

    If you have connection issues with the auto populated FQDN in the FL plan, you can do **one of the following**:

	- OPTION 1: override the auto populated FQDN value with the :code:`-a` flag.

		.. code-block:: shell

			$ fx plan initialize -a aggregator-hostname.internal-domain.com

	- OPTION 2: override the apparent FQDN of the system by setting an FQDN environment variable.

		.. code-block:: shell

			$ export FQDN=x.x.x.x

		and initializing the FL plan

		.. code-block:: shell

			$ fx plan initialize


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

    - Ensure each node in the federation has a valid public key infrastructure (PKI) certificate. See :doc:`../../developer_guide/utilities/pki` for details on available workflows.
    - Distribute the workspace from the aggregator node to the other collaborator nodes.


.. _install_certs_agg:

**On the Aggregator Node:**

Setting Up the Certificate Authority

1. Change to the path of your workspace:

    .. code-block:: shell

       $ cd WORKSPACE_PATH

2. Set up the aggregator node as the `certificate authority <https://en.wikipedia.org/wiki/Certificate_authority>`_ for the federation.

 All certificates will be signed by the aggregator node. Follow the instructions and enter the information as prompted. The command will create a simple database file to keep track of all issued certificates.

    .. code-block:: shell

       $ fx workspace certify

3. Run the aggregator certificate creation command, replacing :code:`AFQDN` with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator node.

    .. code-block:: shell

       $ fx aggregator generate-cert-request --fqdn AFQDN

    .. note::

       On Linux\*\, you can discover the FQDN with this command:

           .. code-block:: shell

              $ hostname --all-fqdns | awk '{print $1}'

   .. note::

      You can override the apparent FQDN by setting it explicitly via the :code:`--fqdn` parameter.

        .. code-block:: shell

            $ fx aggregator generate-cert-request --fqdn AFQDN

      If you omit the :code:`--fdqn` parameter, then :code:`fx` will automatically use the FQDN of the current node assuming the node has been correctly set with a static address.

        .. code-block:: shell

            $ fx aggregator generate-cert-request

4. Run the aggregator certificate signing command, replacing :code:`AFQDN` with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator node.

    .. code-block:: shell

       $ fx aggregator certify --fqdn AFQDN


   .. note::

      You can override the apparent FQDN of the system by setting an FQDN environment variable (:code:`export FQDN=AFQDN`) before signing the certificate.

        .. code-block:: shell

           $ fx aggregator certify --fqdn AFQDN

5. This node now has a signed security certificate as the aggregator for this new federation. You should have the following files.

    +---------------------------+--------------------------------------------------+
    | File Type                 | Filename                                         |
    +===========================+==================================================+
    | Certificate chain         | WORKSPACE.PATH/cert/cert_chain.crt               |
    +---------------------------+--------------------------------------------------+
    | Aggregator certificate    | WORKSPACE.PATH/cert/server/agg_{AFQDN}.crt       |
    +---------------------------+--------------------------------------------------+
    | Aggregator key            | WORKSPACE.PATH/cert/server/agg_{AFQDN}.key       |
    +---------------------------+--------------------------------------------------+

    where **AFQDN** is the fully-qualified domain name of the aggregator node.

.. _workspace_export:

Exporting the Workspace


1. Export the workspace so that it can be imported to the collaborator nodes.

    .. code-block:: shell

       $ fx workspace export

   The :code:`export` command will archive the current workspace (with a :code:`zip` file extension) and create a **requirements.txt** of the current Python\*\ packages in the virtual environment.

2. The next step is to transfer this workspace archive to each collaborator node.


.. _install_certs_colab:

**On the Collaborator Node**:

Importing the Workspace

1. Copy the :ref:`workspace archive <workspace_export>` from the aggregator node to the collaborator nodes.

2. Import the workspace archive.

    .. code-block:: shell

       $ fx workspace import --archive WORKSPACE.zip

 where **WORKSPACE.zip** is the name of the workspace archive. This will unzip the workspace to the current directory and install the required Python packages within the current virtual environment.

3. For each test machine you want to run as collaborator nodes, create a collaborator certificate request to be signed by the certificate authority.

 Replace :code:`COL_LABEL` with the label you assigned to the collaborator. This label does not have to be the FQDN; it can be any unique alphanumeric label.

    .. code-block:: shell

       $ fx collaborator create -n {COL_LABEL} -d {DATA_PATH:optional}
       $ fx collaborator generate-cert-request -n {COL_LABEL}


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

    .. code-block:: shell

       $ fx collaborator certify --request-pkg /PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip

   where :code:`/PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip` is the path to the Collaborator CSR Package containing the :code:`.csr` file from the collaborator node. The certificate authority will sign this certificate for use in the federation.

   The command packages the signed collaborator certificate, along with the **cert_chain.crt** file needed to verify certificate signatures, for transport back to the collaborator node:

    +---------------------------------+------------------------------------------------------------+
    | File Type                       | Filename                                                   |
    +=================================+============================================================+
    | Certificate and Chain Package   | WORKSPACE.PATH/agg_to_col_{COL_LABEL}_signed_cert.zip      |
    +---------------------------------+------------------------------------------------------------+

5. On the collaborator node, import the signed certificate and certificate chain into your workspace.

    .. code-block:: shell

       $ fx collaborator certify --import /PATH/TO/agg_to_col_{COL_LABEL}_signed_cert.zip



.. _running_the_federation.start_nodes:


STEP 3: Start the Federation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**On the Aggregator Node:**

1. Start the Aggregator.

    .. code-block:: shell

       $ fx aggregator start

 Now, the Aggregator is running and waiting for Collaborators to connect.

.. _running_collaborators:

**On the Collaborator Nodes:**

1. Open a new terminal, change the directory to the workspace, and activate the virtual environment.

2. Run the Collaborator.

    .. code-block:: shell

       $ fx collaborator start -n {COLLABORATOR_LABEL}

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


Post Experiment
^^^^^^^^^^^^^^^

Experiment owners may access the final model in its native format.
Among other training artifacts, the aggregator creates the last and best aggregated (highest validation score) model snapshots. One may convert a snapshot to the native format and save the model to disk by calling the following command from the workspace:

.. code-block:: shell

    $ fx model save -i model_protobuf_path.pth -o save_model_path

In order for this command to succeed, the **TaskRunner** used in the experiment must implement a :code:`save_native()` method.

Another way to access the trained model is by calling the API command directly from a Python script:

.. code-block:: python

    from openfl import get_model
    model = get_model(plan_config, cols_config, data_config, model_protobuf_path)

In fact, the :code:`get_model()` method returns a **TaskRunner** object loaded with the chosen model snapshot. Users may utilize the linked model as a regular Python object.


.. _running_the_federation_docker:


Running inside Docker
---------------------

There are two ways you can run OpenFL with Docker\*\.

- `Option 1: Deploy a Federation in a Docker Container`_
- `Option 2: Deploy Your Workspace in a Docker Container`_


.. _running_the_federation_docker_base_image:

Option 1: Deploy a Federation in a Docker Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    You have to built an OpenFL image. See :ref:`installation` for details.


1. Run the OpenFL image.

    .. code-block:: shell

       $ docker run -it --network host openfl


You can now experiment with OpenFL in the container. For example, you can test the project pipeline with the `"Hello Federation" bash script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.sh>`_.


.. _running_the_federation_docker_workspace:

Option 2: Deploy Your Workspace in a Docker Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    You have to set up a TaskRunner and run :code:`fx plan initialize` in the workspace directory. See `STEP 1: Create a Workspace`_ for details.


1. Build an image with the workspace you created.

    .. code-block:: shell

       $ fx workspace dockerize


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

3. Generate public key infrastructure (PKI) certificates for all collaborators and the aggregator. See :doc:`../../developer_guide/utilities/pki` for details.

4. `STEP 3: Start the Federation`_.

.. toctree
..    overview.how_can_intel_protect_federated_learning
..    overview.what_is_intel_federated_learning
