.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

Federated Evaluation
=======================================

Introduction to Federated Evaluation
-------------------------------------

Model evaluation is an essential part of the machine learning development cycle. In a traditional centralized learning system, all evaluation data is collected on a localized server. Because of this, centralized evaluation of machine learning models is a fairly straightforward task. However, in a federated learning system, data is distributed across multiple decentralized devices or nodes. In an effort to preserve the security and privacy of the distributed data, it is infeasible to simply aggregate all the data into a centralized system. Federated evaluation offers a solution by assessing the model at the client side and aggregating the accuracy without ever having to share the data. This is crucial for ensuring the model's effectiveness and reliability in diverse and real-world environments while respecting privacy and data locality. Further sections of this document will detail how Federated Evaluation,a core feature within OpenFL, can be leveraged to achieve decentralized evaluation of an existing model.

OpenFL's Support for Federated Evaluation
------------------------------------------

OpenFL, a flexible framework for Federated Learning, has the capability to perform federated evaluation by modifying the federation plan. In this document, we will show how OpenFL can facilitate this process through its `TaskRunner API <https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html>`_, where the model evaluation is distributed across various collaborators before being sent to the aggregator. For the task runner API, this involves minor modifications to the ``plan.yaml`` file, which defines the workflow and tasks for the federation. In particular, the federation plan should be defined to run for one forward pass and perform only aggregated model validation.

In general a Federated Evaluation pipeline is as follows:

1. **Setup**: Initialize the federation with the modified ``plan.yaml`` set to run for one round and only perform aggregated model validation
2. **Execution**: Run the federation. The model is distributed across collaborators for evaluation.
3. **Evaluation**: Each collaborator evaluates the model on its local data.
4. **Aggregation**: The aggregator collects and aggregates these metrics to assess overall model performance.

Overview
--------
OpenFL now supports FedEval even more seamlessly through its task runner API and the federation plan as defaults have been further refined with both training (a.k.a "learning") and evaluation task_groups being predefined in the default assigner configuration.

In addition, one can now initialize a workspace for evaluation with a pre-trained model load feature thereby eliminating the need to manually replace the ``init.pbuf`` file. 

Further more all the evaluation run overrides like round_number check skipping, defaulting round_number to 1 and task_group selection, which were earlier to be manually ensured by changing the default plan configurations, are all now baked in aggregator behavior and no manual edits are needed to switch between `learning` and `evaluation` run of a plan. Basically the plan is distributed once and can be used for both learning and evaluation.

Requirements
------------
- Latest OpenFL built from source.
- A pre-trained model file (by default named ``best.pbuf``, unless overridden via aggregator configuration) should be available.
- Familiarity with basic OpenFL commands (workspace creation, certificate generation, etc.)

Federated Evaluation using TaskRunner API
----------------------------------------------

This section walks you through an end-to-end example of using the Task Runner API for Federated Evaluation (FedEval) by highlighting  the complete workflow - from training to evaluation.

Workflow Overview
-----------------

**Training Phase:**

- Create a training workspace using the torch/mnist template.

- Certify the workspace and generate certificates for the aggregator and collaborators.

- Initialize the federation plan (default settings target the "learning" task_group).

- Start the aggregator and collaborators in learning mode.

- Save the best model (typically stored as ``best.pbuf`` unless overridden via aggregator configuration).

**Evaluation Phase:**

- Create a new workspace for evaluation using the same torch/mnist template.

- Certify the evaluation workspace and generate the required certificates.

- Initialize the federation plan, loading your pre-trained model using the `-i` flag. For example:

.. code-block:: bash

    fx plan initialize -i ~/trained_model.pbuf

- Start the aggregator in evaluation mode using the `--task_group` flag to override the default behavior:
    
.. code-block:: bash

    fx aggregator start --task_group evaluation
        
- Start your evaluation collaborators and verify that only evaluation tasks are dispatched.

**Aggregator Command Details:**

The aggregator start command supports the optional `--task_group` argument. If this flag is not provided the aggregator will distribute all defined task_groups according to the plan's assigner configuration.

By default, unless overridden by user, the assigner configurations guarantees that only "learning" tasks are assigned.

.. code-block:: shell

    Usage: fx aggregator start [OPTIONS]

    Starts the aggregator service.

    Options:
      -p, --plan PATH             Path to an FL plan.  [default: plan/plan.yaml]
      -c, --authorized_cols PATH  Path to an authorized collaborator list.  [default: plan/cols.yaml]
      --task_group TEXT           Task group to execute as defined in the plan task assigner.
      --help                      Show this message and exit.

**Plan Command Details:**

The plan initialize command supports the optional `--init_model_path` (shortform `-i`) argument. When this option is used and points to a model protobuf file, it will load that model as initial model during plan initialization phase to either further train or evaluate. However in this example we shall use this for evaluation.

.. code-block:: shell

    Usage: fx plan initialize [OPTIONS]

    Initializes a Data Science plan and generates a protobuf file of the initial model weights for the federation.

    Options:
    -p, --plan_config PATH         Path to an FL plan.  [default: plan/plan.yaml]
    -c, --cols_config PATH         Path to an authorized collaborator list.  [default: plan/cols.yaml]
    -d, --data_config PATH         The dataset shard configuration file.  [default: plan/data.yaml]
    -a, --aggregator_address TEXT  The FQDN of the federation agregator
    -f, --input_shape TEXT         The input spec of the model.

                                    May be provided as a list for single input head: ``--input-shape [3,32,32]``,

                                    or as a dictionary for multihead models (must be passed in quotes):

                                    ``--input-shape "{'input_0': [1, 240, 240, 4],'input_1': [1, 240, 240, 1]}"``.
    -g, --gandlf_config TEXT       GaNDLF Configuration File Path
    -r, --install_reqs BOOLEAN     If set, installs packages listed under 'requirements.txt'.  [default: True]
    -i, --init_model_path PATH     Path to initial model protobuf file.
    --help                         Show this message and exit.

The following section ensures that you have full guidance through the tasks required to transition from training (a.k.a learning) into evaluation using the TaskRunner API.

Detailed Instructions
---------------------

**1. Training Phase:** Workspace Setup and Federation Run

Create a training workspace (for example, using the torch/mnist template):

.. code-block:: bash

    fx workspace create --prefix ./cnn_train --template torch/mnist
    cd cnn_train
    fx workspace certify
    fx aggregator generate-cert-request
    fx aggregator certify --silent

Initialize the plan normally:

.. code-block:: bash

    fx plan initialize

By default the assigner ensures that only "learning" task_group tasks are executed

Run the federation using your collaborators. For example:

.. code-block:: bash

    fx collaborator create -n collaborator1 -d 1
    fx collaborator generate-cert-request -n collaborator1
    fx collaborator certify -n collaborator1 --silent

    fx collaborator create -n collaborator2 -d 2
    fx collaborator generate-cert-request -n collaborator2
    fx collaborator certify -n collaborator2 --silent

    fx aggregator start > ~/fx_aggregator.log 2>&1 &
    fx collaborator start -n collaborator1 > ~/collab1.log 2>&1 &
    fx collaborator start -n collaborator2 > ~/collab2.log 2>&1 &

After training is complete, note the best model's performance and save the best model file (``best.pbuf``) as generated in your workspace (e.g. under cnn_train/save/).

In this example run we will save the ``best.pbuf`` in home-directory and name it as ``trained_model.pbuf``

.. code-block:: bash

    cp ./cnn_train/save/best.pbuf ~/trained_model.pbuf


**2. Evaluation Phase:** Workspace Setup Without Manual Plan Changes

Create a new workspace for evaluation using the same template:

.. code-block:: bash

    fx workspace create --prefix ./cnn_eval --template torch/mnist
    cd cnn_eval
    fx workspace certify
    fx aggregator generate-cert-request
    fx aggregator certify --silent

Since the default plan already includes definitions for both "learning" and "evaluation" task_groups, you do not need to modify the round_number or manually edit the assigner section. 

Simply initialize the plan and load your pre-trained model by specifying the `-i` option as shown below :

.. code-block:: bash

    fx plan initialize -i ~/trained_model.pbuf

This command loads the best model from previous training run into the evaluation workspace without any manual file replacement.

**3. Running the Evaluation Federation**

Start the federation for evaluation by explicitly assigning evaluation task_group on aggregator start:

.. code-block:: bash

    # Create evaluation collaborators as before:
    fx collaborator create -n collaborator1 -d 1
    fx collaborator generate-cert-request -n collaborator1
    fx collaborator certify -n collaborator1 --silent

    fx collaborator create -n collaborator2 -d 2
    fx collaborator generate-cert-request -n collaborator2
    fx collaborator certify -n collaborator2 --silent

    # Start the aggregator in evaluation mode:
    fx aggregator start --task_group evaluation > ~/fx_aggregator.log 2>&1 &
    fx collaborator start -n collaborator1 > ~/collab1.log 2>&1 &
    fx collaborator start -n collaborator2 > ~/collab2.log 2>&1 &

With the aggregator running with the "evaluation" task_group (set via `--task_group evaluation`), it will automatically bypass the round_number check and dispatch only the evaluation task i.e 'aggregated_model_validation' to each collaborator. 

Aggregator will further ensure that the evaluation tasks run only for 1 iteration irrespective of the number of rounds of training defined in the plan.

Log messages will indicate that:
   - The aggregator is set to "evaluation" mode.
   - Aggregator will log skipping of round number check and overrides to number of rounds of federation run.
   - Collaborators are receiving only the aggregated model validation task.
   - The final aggregated accuracy is reported matching the pre-trained model's performance

New Features Highlight
----------------------
- **Default Plan Completeness:** Both "learning" and "evaluation" task_groups are pre-defined in the default assigner. No manual edits are necessary.
- **Model Loading via Initialization:** Use the ``fx plan initialize -i`` option to load a pre-trained ``best.pbuf`` model directly during plan initialization.
- **Command-Line Flag for task_group selection:** The ``--task_group`` flag allows the aggregator to switch to any task_group present in the assigner.

Troubleshooting
---------------
- **Plan Initialization:** Ensure that fx plan initialize -i correctly loads your pre-trained model (check the log output).
- **Certificate Validity:** Verify that all certificate generation steps have completed successfully.
- **Log Monitoring:** Use tail on the various logs (plan, aggregator, collaborator) to confirm that only evaluation tasks are being dispatched.
- **Network and TLS:** Confirm that network settings (e.g. aggregator address, port, and TLS configurations) remain consistent between training and evaluation.

Conclusion
----------
FedEval feature being more tightly integrated, one no longer needs to modify the federation plan manually for evaluation rounds nor manually replace the ``init.pbuf`` file.

Simply load the best model using the `-i`` option with `fx plan initialize` and run the aggregator with the `--task_group evaluation` flag.

These improvements simplify switching between learning and evaluation tasks via task_groups and ensure a seamless workflow for federated model assessment.