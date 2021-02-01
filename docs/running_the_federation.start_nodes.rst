.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

***********************
Starting the Federation
***********************

On the Aggregator
~~~~~~~~~~~~~~~~~

1. Now we’re ready to start the aggregator by running the Python script. 

    .. code-block:: console
    
       $ fx aggregator start

    At this point, the aggregator is running and waiting
    for the collaborators to connect. When all of the collaborators
    connect, the aggregator starts training. When the last round of
    training is complete, the aggregator stores the final weights in
    the protobuf file that was specified in the YAML file
    (in this case *save/${WORKSPACE_TEMPLATE}_latest.pbuf*).

.. _running_collaborators:

On the Collaborator
~~~~~~~~~~~~~~~~~~~

1. Open a new terminal, change the directory to the workspace, and activate the virtual environment.

2. Now run the collaborator with the :code:`fx` command.

    .. code-block:: console

       $ fx collaborator start -n COLLABORATOR.LABEL

    where **COLLABORATOR_LABEL** is the label for this collaborator.

    .. note::

       Each workspace may have multiple Federated Learning plans and multiple collaborator lists associated with it.
       Therefore, the Collaborator has the following optional parameters.
       
           +-------------------------+---------------------------------------------------------+
           | Optional Parameters     | Description                                             |
           +=========================+=========================================================+
           | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
           +-------------------------+---------------------------------------------------------+
           | -d, --data_config PATH  | The data set/shard configuration file                   |
           +-------------------------+---------------------------------------------------------+

3. Repeat this for each collaborator in the federation. Once all collaborators have joined,  the aggregator will start and you will see log messages describing the progress of the federated training.
