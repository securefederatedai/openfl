.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_with_gandlf:

Federated Learning using GaNDLF
===============================

This guide will show you how to take an existing model using the `Generally Nuanced Deep Learning Framework (GaNDLF) <https://github.com/mlcommons/GaNDLF>`_ experiment to a federated environment. 


`Aggregator-Based Workflow`_
    Define an experiment and distribute it manually. All participants can verify model code and `FL plan <https://openfl.readthedocs.io/en/latest/running_the_federation.html#federated-learning-plan-fl-plan-settings>`_ prior to executing the code/model. The federation is terminated when the experiment is finished, and appropriate statistics are generated.


.. _running_the_federation_aggregator_based_gandlf:

TaskRunner API
=========================

An overview of this workflow is shown below.

.. figure:: ../images/openfl_flow.png

.. centered:: Overview of the TaskRunner API


This method uses short-lived components in a federation, which is terminated when the experiment is finished. The components are as follows:

- The *Collaborator* uses a local dataset to train a global model and sends the model updates to the *Aggregator*, which aggregates them to create the new global model.
- The *Aggregator* is framework-agnostic, while the *Collaborator* can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_\* \  or `PyTorch <https://pytorch.org/>`_\*\. `GaNDLF <https://github.com/mlcommons/GaNDLF>`_ provides a straightforward way to define complete model training pipelines for healthcare data, and is directly compatible with OpenFL.

This guide will demonstrate how to take an existing `GaNDLF model configuration <https://mlcommons.github.io/GaNDLF/getting_started/>`_ (e.g., for segmentation), embed this within the Federated Learning plan (FL plan) along with the Python\*\  code that defines the model and the data loader. The FL plan is a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file that defines the collaborators, aggregator, connections, models, data, and any other parameters that describe the training.


.. _plan_settings_gandlf:



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


.. _running_the_federation_manual_gandlf:


.. _simulate_a_federation_gandlf:


Simulate a federation
---------------------

.. note::

    Ensure you have installed the OpenFL package on every node (aggregator and collaborators) in the federation.

    See :ref:`installation` for details.


You can use the `"Hello Federation" bash script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.py>`_ to quickly create a federation (an aggregator node and two collaborator nodes) to test the project pipeline.

.. literalinclude:: ../../tests/github/test_hello_federation.py
  :language: bash

However, continue with the following procedure for details in creating a federation with an aggregator-based workflow.

    `STEP 1: Install GaNDLF prerequisites and Create a Workspace`_


        - Creates a federated learning workspace on one of the nodes.


    `STEP 2: Configure the Federation`_

        - Ensures each node in the federation has a valid public key infrastructure (PKI) certificate.
        - Distributes the workspace from the aggregator node to the other collaborator nodes.


    `STEP 3: Start the Federation`_


.. _creating_workspaces_gandlf:


STEP 1: Install GaNDLF prerequisites and Create a Workspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.	Start a Python 3.9 (>=3.9, <3.12) virtual environment and confirm OpenFL is available.

	.. code-block:: python

		fx


2.     `Install GaNDLF from sources <https://mlcommons.github.io/GaNDLF/setup/#install-from-sources>`_ (if not already).

3.     Create GaNDLF's Data CSVs. The example below is for 3D Segmentation using the unit test data:

    .. code-block:: shell
        
        $ python -c "from testing.test_full import test_generic_download_data, test_generic_constructTrainingCSV; test_generic_download_data(); test_generic_constructTrainingCSV()"
        # Creates training CSV
        $ head -n 8 testing/data/train_3d_rad_segmentation.csv > train.csv
        $ head -n 1 testing/data/train_3d_rad_segmentation.csv > val.csv
        # Creates validation CSV
        $ tail -n +9 testing/data/train_3d_rad_segmentation.csv >> val.csv

    .. note::

       This step creates sample data CSVs for this tutorial. In a real federation, you should bring your own Data CSV files from GaNDLF that reflect the data present on your system 


4. 	Use the :code:`gandlf_seg_test` template

	Set the environment variables to use the :code:`gandlf_seg_test` as the template and :code:`${HOME}/my_federation` as the path to the workspace directory.

    .. code-block:: shell

        $ export WORKSPACE_TEMPLATE=gandlf_seg_test
        $ export WORKSPACE_PATH=${HOME}/my_federation

4.  Create a workspace directory for the new federation project.

    .. code-block:: shell

       $ fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}


5.  Change to the workspace directory.

    .. code-block:: shell

        $ cd ${WORKSPACE_PATH}

6.  Copy the GaNDLF Data CSVs into the default path for model initialization

     .. code-block:: shell

        # 'one' is the default name of the first collaborator in `plan/data.yaml`. 
        $ mkdir -p data/one
        $ cp ~/GaNDLF/train.csv data/one
        $ cp ~/GaNDLF/val.csv data/one

6.  Create the FL plan and initialialize the model weights.


    This step will initialize the federated learning plan and initialize the random model weights that will be used by all collaborators at the start of the expeirment.

    .. code-block:: shell

	$ fx plan initialize

    Alternatively, to use your own GaNDLF configuration file, you can import it into the plan with the following command:

    .. code-block:: shell

	$ fx plan initialize --gandlf_config ${PATH_TO_GANDLF_CONFIG}.yaml


    The FL plan is described by the **plan.yaml** file located in the **plan** directory of the workspace. OpenFL aims to make it as easy as possible to take an existing GaNDLF experiment and make it run across a federation. 
    
    Each YAML top-level section contains the following subsections:
    
    - ``template``: The name of the class including top-level packages names. An instance of this class is created when the plan gets initialized.
    - ``settings``: The arguments that are passed to the class constructor.
    - ``defaults``: The file that contains default settings for this subsection.
      Any setting from defaults file can be overridden in the **plan.yaml** file.
    
    The following is an example of the GaNDLF Segmentation Test **plan.yaml**. Notice the **task_runner/settings/gandlf_config** block where the GaNDLF configuration file is embedded:
    
    .. literalinclude:: ../../openfl-workspace/gandlf_seg_test/plan/plan.yaml
      :language: yaml


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



.. _configure_the_federation_gandlf:


STEP 2: Configure the Federation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The objectives in this step:

    - Ensure each node in the federation has a valid public key infrastructure (PKI) certificate. See :doc:`utilities/pki` for details on available workflows.
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

      You can override the apparent FQDN of the system by setting an FQDN environment variable before creating the certificate.

        .. code-block:: shell

            $ export FQDN=x.x.x.x
            $ fx aggregator generate-cert-request 

      If you omit the :code:`--fdqn` parameter, then :code:`fx` will automatically use the FQDN of the current node assuming the node has been correctly set with a static address.

        .. code-block:: shell

            $ fx aggregator generate-cert-request

4. Run the aggregator certificate signing command, replacing :code:`AFQDN` with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator node.

    .. code-block:: shell

       $ fx aggregator certify


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

2.     `Install GaNDLF from sources <https://mlcommons.github.io/GaNDLF/setup/#install-from-sources>`_ (if not already).

3. Import the workspace archive.

    .. code-block:: shell

       $ fx workspace import --archive WORKSPACE.zip

 where **WORKSPACE.zip** is the name of the workspace archive. This will unzip the workspace to the current directory and install the required Python packages within the current virtual environment.

4. For each test machine you want to run as collaborator nodes, create a collaborator certificate request to be signed by the certificate authority.

 Replace :code:`COL_LABEL` with the label you assigned to the collaborator. This label does not have to be the FQDN; it can be any unique alphanumeric label.

    .. code-block:: shell

       $ fx collaborator generate-cert-request -n {COL_LABEL} -d data/{COL_LABEL}


The creation script will specify the path to the data. In this case, the GaNDLF Data Loader will look for **train.csv** and **valid.csv** at the path that's provided. Before running the experiment, you will need to configure both train.csv and valid.csv **manually for each collaborator** so that each collaborator has the correct datasets. For example, if the collaborator's name is `one`, collaborator one would load :code:`data/one/train.csv` and :code:`data/one/valid.csv` at experiment runtime, and collaborator two would load :code:`data/two/train.csv` and :code:`data/two/valid.csv`. 

 This command will also create the following files:

    +-----------------------------+--------------------------------------------------------+
    | File Type                   | Filename                                               |
    +=============================+========================================================+
    | Collaborator CSR            | WORKSPACE.PATH/cert/client/col_{COL_LABEL}.csr         |
    +-----------------------------+--------------------------------------------------------+
    | Collaborator key            | WORKSPACE.PATH/cert/client/col_{COL_LABEL}.key         |
    +-----------------------------+--------------------------------------------------------+
    | Collaborator CSR Package    | WORKSPACE.PATH/col_{COL_LABEL}_to_agg_cert_request.zip |
    +-----------------------------+--------------------------------------------------------+

5. Copy/scp the WORKSPACE.PATH/col_{COL_LABEL}_to_agg_cert_request.zip file to the aggregator node (or local workspace if using the same system)

    .. code-block:: shell

       $ scp WORKSPACE.PATH/col_{COL_LABEL}_to_agg_cert_request.zip AGGREGATOR_NODE:WORKSPACE_PATH/


6. On the aggregator node (i.e., the certificate authority in this example), sign the Collaborator CSR Package from the collaborator nodes.

    .. code-block:: shell

       $ fx collaborator certify --request-pkg /PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip

   where :code:`/PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip` is the path to the Collaborator CSR Package containing the :code:`.csr` file from the collaborator node. The certificate authority will sign this certificate for use in the federation.

   The command packages the signed collaborator certificate, along with the **cert_chain.crt** file needed to verify certificate signatures, for transport back to the collaborator node:

    +---------------------------------+------------------------------------------------------------+
    | File Type                       | Filename                                                   |
    +=================================+============================================================+
    | Certificate and Chain Package   | WORKSPACE.PATH/agg_to_col_{COL_LABEL}_signed_cert.zip      |
    +---------------------------------+------------------------------------------------------------+

7. Copy/scp the WORKSPACE.PATH/agg_to_col_{COL_LABEL}_signed_cert.zip file to the collaborator node (or local workspace if using the same system)

    .. code-block:: shell

       $ scp WORKSPACE.PATH/agg_to_col_{COL_LABEL}_signed_cert.zip COLLABORATOR_NODE:WORKSPACE_PATH/


8. On the collaborator node, import the signed certificate and certificate chain into your workspace.

    .. code-block:: shell

       $ fx collaborator certify --import /PATH/TO/agg_to_col_{COL_LABEL}_signed_cert.zip



.. _running_the_federation.start_nodes.gandlf:


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

Experiment owners may access the final model in its native format. Once the model has been converted to its native format, inference can be done using `GaNDLF's inference API <https://mlcommons.github.io/GaNDLF/usage/#running-gandlf-traininginference>`_.
Among other training artifacts, the aggregator creates the last and best aggregated (highest validation score) model snapshots. One may convert a snapshot to the native format and save the model to disk by calling the following command from the workspace:

.. code-block:: shell

    $ fx model save -i model_protobuf_path.pth -o save_model_path

In order for this command to succeed, the **TaskRunner** used in the experiment must implement a :code:`save_native()` method.

Another way to access the trained model is by calling the API command directly from a Python script:

.. code-block:: python

    from openfl import get_model
    model = get_model(plan_config, cols_config, data_config, model_protobuf_path)

In fact, the :code:`get_model()` method returns a **TaskRunner** object loaded with the chosen model snapshot. Users may utilize the linked model as a regular Python object.

Running Inference with GaNDLF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that you have generated the final federated model in pytorch format, you can use the model by following the `GaNDLF inference instructions <https://mlcommons.github.io/GaNDLF/usage/#inference>`_
