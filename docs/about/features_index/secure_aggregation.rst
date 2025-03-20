.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

Secure Aggregation
=======================================

In Federated Learning (FL), Secure Aggregation (SecAgg) restricts the aggregator to only learn the summation or average of the updates from collaborators.

OpenFL integrates `SecAgg <https://eprint.iacr.org/2017/281.pdf>`_ into TaskRunner API as well as the Workflow API.

TaskRunner API
-------------------------------------

OpenFL treats SecAgg as a core security feature and can be enabled for any experiment by simply modifying the plan.
**NOTE**: 
- `pycryptodome <https://pypi.org/project/pycryptodome/>`_ is a required dependency that must be installed on the participant nodes before starting the experiment.
- Secure aggregation only supports `WaitForAllPolicy` as `straggler_handling_policy`; the support for other policies will be addressed in future updates.
- The impact of secure aggregation on the aggregator and/or collaborator restart will be assessed, and additional resiliency features will be introduced in subsequent updates.

The following plan shows secure aggregation being enabled on `keras/mnist <https://github.com/securefederatedai/openfl/tree/develop/openfl-workspace/keras/mnist>`_ workspace by simply modifying the plan.

.. code-block:: yaml
    :emphasize-lines: 10,51,52,53

    aggregator:
        settings:
            best_state_path: save/best.pbuf
            db_store_rounds: 2
            init_state_path: save/init.pbuf
            last_state_path: save/last.pbuf
            persist_checkpoint: true
            persistent_db_path: local_state/tensor.db
            rounds_to_train: 1
            secure_aggregation: true
        template: openfl.component.Aggregator
    assigner:
        settings:
            task_groups:
            - name: learning
            percentage: 1.0
            tasks:
            - aggregated_model_validation
            - train
            - locally_tuned_model_validation
            - name: evaluation
            percentage: 0
            tasks:
            - aggregated_model_validation
        template: openfl.component.RandomGroupedAssigner
    collaborator:
        settings:
            db_store_rounds: 1
            use_delta_updates: false
            opt_treatment: RESET
        template: openfl.component.Collaborator
    compression_pipeline:
        settings: {}
        template: openfl.pipelines.NoCompressionPipeline
    data_loader:
        settings:
            batch_size: 256
            collaborator_count: 2
            data_group_name: mnist
        template: src.dataloader.KerasMNISTInMemory
    network:
        settings:
            agg_addr: localhost
            agg_port: 53788
            cert_folder: cert
            client_reconnect_interval: 5
            hash_salt: auto
            require_client_auth: true
            use_tls: true
        template: openfl.federation.Network
    straggler_handling_policy:
        settings: {}
        template: openfl.component.aggregator.straggler_handling.WaitForAllPolicy
    task_runner:
        settings: {}
        template: src.taskrunner.KerasCNN
    tasks:
        aggregated_model_validation:
            function: validate_task
            kwargs:
            apply: global
            batch_size: 32
            metrics:
            - accuracy
        locally_tuned_model_validation:
            function: validate_task
            kwargs:
            apply: local
            batch_size: 32
            metrics:
            - accuracy
        settings: {}
        train:
            function: train_task
            kwargs:
            batch_size: 32
            epochs: 1
            metrics:
            - loss
        
As can be seen in the above plan, by only enabling ``aggregator.settings.secure_aggregation`` in the workspace plan, one can enable SecAgg.

After the flags have been set in plan.yml and the setup for the experiment is completed, one can verify that SecAgg was enabled by looking at the aggregator logs

.. code-block:: bash

    [21:55:01] INFO     SecAgg: recreated secrets successfully                                                                                          setup.py:281
               INFO     SecAgg: setup completed, saved required tensors to db

Similarly, in the collaborator logs

.. code-block:: bash

               INFO     Secure aggregation is enabled, starting setup...                                                                    secure_aggregation.py:48
    [21:55:01] INFO     SecAgg: setup completed, saved required tensors to db.


Workflow API
-------------------------------------

OpenFL provides `utility functions <https://github.com/securefederatedai/openfl/tree/develop/openfl/utilities/secagg>`_ that can be utilised to perform SecAgg in Workflow API.

An example notebook can be found `here <https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/SecAgg>`_ that showcases how the secure aggregation flow can be achieved in Workflow API using both, LocalRuntime and FederatedRuntime.