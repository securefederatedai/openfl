.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation.start_nodes:

****************************
STEP 3: Start the Federation
****************************

On the Aggregator Node
======================

1. Start the aggregator. 

    .. code-block:: console
    
       fx aggregator start

    Now, the aggregator is running and waiting for the collaborators to connect. When all of the collaborators connect, the aggregator starts training. When the last round of training is completed, the aggregator stores the final weights in the protobuf file that was specified in the YAML file, which in this example is located at *save/${WORKSPACE_TEMPLATE}_latest.pbuf*.

.. _running_collaborators:

On the Collaborator Nodes
=========================

1. Open a new terminal, change the directory to the workspace, and activate the virtual environment.

2. Run the collaborator.

    .. code-block:: console

       fx collaborator start -n COLLABORATOR.LABEL

    where :code:`COLLABORATOR_LABEL` is the label for this collaborator.

    .. note::

       Each workspace may have multiple FL Plans and multiple collaborator lists associated with it.
       Therefore, the collaborator has the following optional parameters.
       
           +-------------------------+---------------------------------------------------------+
           | Optional Parameters     | Description                                             |
           +=========================+=========================================================+
           | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
           +-------------------------+---------------------------------------------------------+
           | -d, --data_config PATH  | The data set/shard configuration file                   |
           +-------------------------+---------------------------------------------------------+

3. Repeat the earlier steps for each collaborator node in the federation. Once all collaborators have connected, the aggregator will start training. You will see log messages describing the progress of the federated training.
