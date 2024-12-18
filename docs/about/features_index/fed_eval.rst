.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

Federated Evaluation
=======================================

Introduction to Federated Evaluation
-------------------------------------

Model evaluation is an essential part of the machine learning development cycle. In a traditional centralized learning system, all evaluation data is collected on a localized server. Because of this, centralized evaluation of machine learning models is a fairly straightforward task. However, in a federated learning system, data is distributed across multiple decentralized devices or nodes. In an effort to preserve the security and privacy of the distributed data, it is infeasible to simply aggregate all the data into a centralized system. Federated evaluation offers a solution by assessing the model at the client side and aggregating the accuracy without ever having to share the data. This is crucial for ensuring the model's effectiveness and reliability in diverse and real-world environments while respecting privacy and data locality

OpenFL's Support for Federated Evaluation
-------------------------------------------------

OpenFL, a flexible framework for Federated Learning, has the capability to perform federated evaluation by modifying the federation plan. In this document, we will show how OpenFL can facilitate this process through its task runner API (aggregator-based workflow), where the model evaluation is distributed across various collaborators before being sent to the aggregator. For the task runner API, this involves minor modifications to the ``plan.yaml`` file, which defines the workflow and tasks for the federation. In particular, the federation plan should be defined to run for one forward pass and perform only aggregated model validation

In general pipeline is as follows:

1. **Setup**: Initialize the federation with the modified ``plan.yaml`` set to run for one round and only perform aggregated model validation
2. **Execution**: Run the federation. The model is distributed across collaborators for evaluation.
3. **Evaluation**: Each collaborator evaluates the model on its local data.
4. **Aggregation**: The aggregator collects and aggregates these metrics to assess overall model performance.

Example Using the Task Runner API (Aggregator-based Workflow)
--------------------------------------------------------------

To demonstrate usage of the task runner API (aggregator-based workflow) for federated evaluation, consider the `Hello Federation example <https://github.com/securefederatedai/openfl/blob/develop/tests/github/test_hello_federation.py>`_. This sample script creates a simple federation with two collaborator nodes and one aggregator node, and executes based on a user specified workspace template. We provide a ``torch_cnn_mnist_fed_eval`` template, which is a federated evaluation template adapted from ``torch_cnn_mnist``. 

This script can be directly executed as follows:

.. code-block:: shell

    $ python test_hello_federation.py --template torch_cnn_mnist_fed_eval
    
In order to adapt this template for federated evaluation, the following defaults were added for assigner, aggregator and tasks and same referenced in the ``plan.yaml``:

.. literalinclude:: ../../../openfl-workspace/torch_cnn_mnist_fed_eval/plan/plan.yaml

.. literalinclude:: ../../../openfl-workspace/workspace/plan/defaults/federated-evaluation/aggregator.yaml

.. literalinclude:: ../../../openfl-workspace/workspace/plan/defaults/federated-evaluation/assigner.yaml

.. literalinclude:: ../../../openfl-workspace/workspace/plan/defaults/federated-evaluation/tasks_torch.yaml

Key Changes for Federated Evaluation by baking in defaults for:

1. **aggregator.settings.rounds_to_train**: Set to 1
2. **assigner**: Assign to aggregated_model_validation instead of default assignments
3. **tasks**: Set to aggregated_model_validation instead of default tasks

**Optional**: modify ``src/pt_cnn.py`` to remove optimizer initialization and definition of loss function as these are not needed for evaluation

This sample script will create a federation based on the `torch_cnn_mnist_fed_eval` template using the `plan.yaml` file defined above, spawning two collaborator nodes and a single aggregator node. The model will be sent to the two collaborator nodes, where each collaborator will perform model validation on its own local data. The accuracy from this model validation will then be send back to the aggregator where it will aggregated into a final accuracy metric. The federation will then be shutdown.

---

Congratulations, you have successfully performed federated evaluation across two decentralized collaborator nodes with minor default reference changes to plan