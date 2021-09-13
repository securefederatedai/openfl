.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_baremetal:

*********************
Create the Federation
*********************

Prerequisites
=============

You have installed the |prodName| package on every node (aggregator and collaborators) in the federation.

See :ref:`install_package` for details.

TL;DR
=====

Here's the :download:`"Hello Federation" bash script <../tests/github/test_hello_federation.sh>` used for testing the project pipeline.

.. literalinclude:: ../tests/github/test_hello_federation.sh
  :language: bash


.. _creating_workspaces:

Create a Workspace on the Aggregator
====================================

1. Start a Python\* \  3.6 (or higher) virtual environment and confirm |productName| is available.

	.. code-block:: python

		fx


2. Choose a workspace template, which are end-to-end federated learning training demonstrations. The following are existing templates:

 - :code:`keras_cnn_mnist`: workspace with a simple `Keras <http://keras.io/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`tf_2dunet`: workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will use the `BraTS <https://www.med.upenn.edu/sbia/brats2017/data.html>`_ dataset and train in a federation.
 - :code:`tf_cnn_histology`: workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_histology`: workspace with a simple `PyTorch <http://pytorch.org/>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_mnist`: workspace with a simple `PyTorch <http://pytorch.org>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.

3.  This example uses the :code:`keras_cnn_mnist` template. 

	Set the environment variables to use the :code:`keras_cnn_mnist` as the template and :code:`${HOME}/my_federation` as the path to the workspace directory.

    .. code-block:: console
    
        $ export WORKSPACE_TEMPLATE=keras_cnn_mnist
        $ export WORKSPACE_PATH=${HOME}/my_federation

    
4.  Create a workspace for the new federation project.

    .. code-block:: console
    
       $ fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}
       
    where :code:`--prefix` is the directory to create your workspace.  
    
    The list of pre-created templates can be found by simply running the command:

    .. code-block:: console
    
       $ fx workspace create --prefix ${WORKSPACE_PATH}
       
    .. note::
    
		You can use your own models by overwriting the Python scripts in the :code:`src` subdirectory in the workspace.

5.  Change to the workspace directory.

    .. code-block:: console
    
        $ cd ${WORKSPACE_PATH}

6.  Install workspace requirements:

    .. code-block:: console
    
        $ pip install -r requirements.txt
      
  
7.  Create an initial set of random model weights.

    .. code-block:: console
    
       $ fx plan initialize


.. note::
	
	While models can be trained from scratch, in many cases the federation performs fine-tuning of a previously trained model. For this reason, pre-trained weights for the model are stored in protobuf files on the aggregator and passed to collaborators during initialization. The protobuf file with the initial weights is found in **${WORKSPACE_TEMPLATE}_init.pbuf**.


    This command initializes the FL Plan and autopopulates the `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ of the aggregator node. This FQDN is embedded within the FL Plan so the collaborators know address of the externally accessible aggregator server to connect to. If you face connection issues with the autopopulated FQDN in the FL Plan, you can do one of the following:
	
	- override the autopopulated FQDN value with the :code:`-a` flag
	
		.. code-block:: console
			$ fx plan initialize -a aggregator-hostname.internal-domain.com
		
	- override the apparent FQDN of the system by setting an FQDN environment variable,

		.. code-block:: console
			$ export FQDN=x.x.x.x
		
	and initializing the FL Plan
	
		.. code-block:: console
			$ fx plan initialize

   
.. note::
    
       Each workspace may have multiple Federated Learning plans and multiple collaborator lists associated with it. Therefore, the Aggregator has the following optional parameters.
       
       +-------------------------+---------------------------------------------------------+
       | Optional Parameters     | Description                                             |
       +=========================+=========================================================+
       | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
       +-------------------------+---------------------------------------------------------+
       | -c, --cols_config PATH  | Authorized collaborator list [default = plan/cols.yaml] |
       +-------------------------+---------------------------------------------------------+
       | -d, --data_config PATH  | The data set/shard configuration file                   |
       +-------------------------+---------------------------------------------------------+    
