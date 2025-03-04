.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _openfl_components:

*****************************
Core Components
*****************************

Open Federated Learning (OpenFL) has the following components:

    - :ref:`openfl_short_lived_components`

.. _openfl_short_lived_components:

Short-Lived Components
======================

These components are terminated when the experiment is finished.
	
    - The *Aggregator* which receives model updates from *Collaborators* and combines them to form the global model.
    - The *Collaborator* which uses local dataset to train a global model.

The *Aggregator* is framework-agnostic, as it operates tensors in OpenFL inner representation,
while the *Collaborator* can use deep learning frameworks as computational backend, such as `TensorFlow* <https://www.tensorflow.org/>`_ or `PyTorch* <https://pytorch.org/>`_.


Aggregator
----------

The Aggregator is a short-lived entity, which means that its lifespan is limited by the experiment execution time.
It orchestrates Collaborators according to the FL plan, performs model aggregation at the end of each round,
and acts as a parameter server for collaborators.

Model weight aggregation logic may be customized via :ref:`plugin mechanism <overriding_agg_fn>`.

The Aggregator is spawned by the :ref:`Director <openfl_ll_components_director>` when a new experiment is submitted.


Collaborator
------------

The Collaborator is a short-lived entity that manages training the model on local data, which includes

    - executing assigned tasks,
    - converting deep learning framework-specific tensor objects to OpenFL inner representation, and
    - exchanging model parameters with the Aggregator.

The Collaborator is created by the :ref:`Envoy <openfl_ll_components_envoy>` when a new experiment is submitted
in the :ref:`Director-based workflow <running_interactive>` (Deprecated). The Collaborator should be started from CLI if a user follows the
:ref:`Aggregator-based workflow <running_the_task_runner>`

Every Collaborator is a unique service. The data loader is loaded with a local *shard descriptor* to perform tasks
included in an FL experiment. At the end of the training task, weight tensors are extracted and sent to the central node
and aggregated.

Converting tensor objects is handled by :ref:`framework adapter <framework_adapter>` plugins.
Included in OpenFL are framework adapters for PyTorch and TensorFlow 2.x.
The list of framework adapters is extensible. User can contribute new framework adapters for deep learning frameworks
they would like see supported in OpenFL.
