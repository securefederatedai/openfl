.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

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

       fx collaborator start -n COLLABORATOR.LABEL

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

