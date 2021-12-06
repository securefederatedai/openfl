.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _creating_workspaces:

********************************************
STEP 1: Create a Workspace on the Aggregator
********************************************

1.	Start a Python\* \  3.6 (or higher) virtual environment and confirm |productName| is available.

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
