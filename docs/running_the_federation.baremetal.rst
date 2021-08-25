.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_baremetal:

Creating the Federation
#######################

TL;DR
~~~~~

Here's the :download:`"Hello Federation" bash script <../tests/github/test_hello_federation.sh>` used for testing the project pipeline.

.. literalinclude:: ../tests/github/test_hello_federation.sh
  :language: bash


Hello Federation - Your First Federated Learning Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will show you how to set up |productName|. 
Before you run the federation make sure you have installed |productName| 
:ref:`using these instructions <install_initial_steps>` on every node (*i.e.* aggregator and collaborators).

.. _creating_workspaces:

On the Aggregator
~~~~~~~~~~~~~~~~~

1. Make sure you have initialized the virtual environment and can run the :code:`fx` command.


2. Choose a workspace template from which to start. Templates are end-to-end federated learning training demos. The existing templates are:

 - :code:`keras_cnn_mnist`: workspace with a simple `Keras <http://keras.io/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`tf_2dunet`: workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will use the `BraTS <https://www.med.upenn.edu/sbia/brats2017/data.html>`_ dataset and train in a federation.
 - :code:`tf_cnn_histology`: workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_histology`: workspace with a simple `PyTorch <http://pytorch.org/>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
 - :code:`torch_cnn_mnist`: workspace with a simple `PyTorch <http://pytorch.org>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.

3.  For this example, we will use the template called :code:`keras_cnn_mnist` and create it within a new directory called :code:`${HOME}/my_federation`.

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
    
    You can use your own models by overwriting the Python scripts in 
    :code:`src` subdirectory in the workspace.

5.  Change to the workspace directory.

    .. code-block:: console
    
        $ cd ${WORKSPACE_PATH}

6.  Install workspace requirements:

    .. code-block:: console
    
        $ pip install -r requirements.txt
      
  
7.  Although it is possible to train models from scratch, it is assumed that in many cases the federation may perform fine-tuning of a previously-trained model. For this reason, the pre-trained weights for the model will be stored within protobuf files on the aggregator and passed to the collaborators during initialization. As seen in the YAML file, the protobuf file with the initial weights is expected to be found in the file **${WORKSPACE_TEMPLATE}_init.pbuf**. For this example, however, we’ll just create an initial set of random model weights and putting it into that file by running the command:

    .. code-block:: console
    
       $ fx plan initialize

    This will initialize the plan and autopopulate the `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ of the aggregator node. This FQDN is embedded within the plan so the collaborators know the externally accessible aggregator server address to connect to. If you face connection issues with the autopopulated FQDN in the plan, this value can be overridden with the :code:`-a` flag, for example :code:`fx plan initialize -a aggregator-hostname.internal-domain.com`. Alternatively, you can override the apparent FQDN of the system by setting an FQDN environment variable (:code:`export FQDN=x.x.x.x`) before invoking :code:`fx plan initialize`.
   
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
