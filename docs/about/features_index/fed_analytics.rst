.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

Federated Analytics
=======================================

Introduction to Federated Analytics
-------------------------------------

Federated Analytics is a privacy-preserving approach to compute statistics or perform data analysis on distributed datasets without aggregating raw data into a centralized location. This method ensures data security while enabling insights to be drawn from decentralized data sources. For instance, one can compute the mean, frequency distributions, or other statistical measures across datasets located on multiple devices. Federated Analytics is particularly valuable in scenarios where data sharing is restricted due to privacy concerns or regulatory constraints.

OpenFL's Support for Federated Analytics
------------------------------------------

OpenFL, a flexible framework for Federated Learning, extends its capabilities to support Federated Analytics. By leveraging the federation plan and task runner API, OpenFL enables users to perform analytics tasks across collaborators. These tasks are defined in the ``plan.yaml`` file and distributed to collaborators for execution. The results are then aggregated by the aggregator to provide global insights.


Example Workspace: Histogram Calculation using sklearn IRIS Dataset
------------------------------------------------------------------------------

The Federated Analytics workspace for histogram calculation demonstrates how to compute frequency distributions of specific features across distributed datasets. This workspace leverages the OpenFL framework to ensure privacy-preserving analytics while providing global insights into the data.

**Task Configuration:**

The analytics tasks are defined in the `plan.yaml` file. For example:

.. code-block:: yaml
    :emphasize-lines: 6,41,43,45

    aggregator:
      defaults: plan/defaults/aggregator.yaml
      template: openfl.component.Aggregator
      settings:
        last_state_path: save/result.json
        rounds_to_train: 1 # Number of training rounds (set to 1 for Federated Analytics).

    collaborator:
      defaults: plan/defaults/collaborator.yaml
      template: openfl.component.Collaborator
      settings:
        use_delta_updates: false
        opt_treatment: RESET

    data_loader:
      defaults: plan/defaults/data_loader.yaml
      template: src.dataloader.IRISInMemory
      settings:
        collaborator_count: 2
        data_group_name: iris
        batch_size: 150

    task_runner:
      defaults: plan/defaults/task_runner.yaml
      template: src.taskrunner.IrisHistogram

    network:
      defaults: plan/defaults/network.yaml

    assigner:
      template: openfl.component.RandomGroupedAssigner
      settings:
        task_groups:
          - name: analytics
            percentage: 1.0
            tasks:
              - analytics

    tasks:
      analytics:
        function: analytics
        aggregation_type:
          template: src.aggregatehistogram.AggregateHistogram
        kwargs:
          columns: ['sepal length (cm)', 'sepal width (cm)']

**Note:** The `function` and `aggregation_type.template` fields in the configuration can be replaced with custom implementations to suit specific use cases. This flexibility allows users to define their own analytics logic and aggregation methods tailored to their requirements.

**Data Distribution**: The dataset is distributed across collaborators, with each collaborator holding a local shard of the data.

**Local Computation**: Each collaborator computes the histogram for the specified feature(s) on its local data shard. This ensures that raw data never leaves the collaborator's environment.

**Aggregation**: The aggregator collects the histograms from all collaborators and combines them to compute the global histogram. The aggregated results are saved in `save/result.json`. This file provides a global view of the frequency distribution for the selected feature, computed in a privacy-preserving manner.


By following this structured approach, the Federated Analytics workspace enables secure and efficient computation of histograms across distributed datasets.

Detailed Instructions
---------------------

Workspace Setup and Federation Run

Create a workspace for analytics (for example, using the federated_analytics/histogram template):

.. code-block:: bash

    fx workspace create --prefix ./analytics_workspace --template federated_analytics/histogram
    cd analytics_workspace
    fx workspace certify
    fx aggregator generate-cert-request
    fx aggregator certify --silent

Initialize the plan normally:

.. code-block:: bash

    fx plan initialize

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

Once the federation run is complete, the results will be saved.

The result file `save/result.json` contains the aggregated histogram data. For example:

.. code-block:: json

    {
        "sepal length (cm) histogram": [
            0.0,
            0.0,
            9.0,
            50.0,
            56.0,
            28.0,
            7.0,
            0.0,
            0.0
        ],
        "sepal length (cm) bins": [
            4.0,
            5.777777671813965,
            7.55555534362793,
            9.333333015441895,
            11.11111068725586,
            12.88888931274414,
            14.666666984558105,
            16.44444465637207,
            18.22222137451172,
            20.0
        ],
        "sepal width (cm) histogram": [
            47.0,
            91.0,
            12.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "sepal width (cm) bins": [
            4.0,
            5.777777671813965,
            7.55555534362793,
            9.333333015441895,
            11.11111068725586,
            12.88888931274414,
            14.666666984558105,
            16.44444465637207,
            18.22222137451172,
            20.0
        ]
    }


Conclusion
----------
Federated Analytics in OpenFL enables privacy-preserving data analysis on distributed datasets. By leveraging the task runner API and predefined analytics tasks, users can seamlessly compute global statistics without compromising data privacy. This feature simplifies the workflow for distributed data analysis and ensures compliance with privacy regulations.