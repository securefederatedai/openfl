.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_aggregator_based:

*************************
Aggregator-Based Workflow
*************************

There are two ways to run this workflow:

- :ref:`Bare metal approach <running_the_federation_manual>`
- :ref:`Docker approach <running_the_federation_docker>`



This workflow uses short-lived components in a federation, which is terminated when the experiment is finished. The components are as follows:

- The *Collaborator* uses a local dataset to train a global model and the *Aggregator* receives model updates from *Collaborators* and combines them to form the global model.
- The *Aggregator* is framework-agnostic, while the *Collaborator* can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_\* \  or `PyTorch <https://pytorch.org/>`_\*\.

.. note::
    For this workflow, you modify the federation workspace to your requirements by editing the Federated Learning plan (FL plan) along with the Python\*\  code that defines the model and the data loader. The FL plan is a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file that defines the collaborators, aggregator, connections, models, data, and any other parameters that describe the training.
    
    See :doc:`../../plan_settings` for details.

An overview of this workflow is shown below.

.. figure:: /images/openfl_flow.png

.. centered:: Overview of the Aggregator-Based Workflow







.. toctree::
   :maxdepth: 3
   :hidden:

   running_the_federation.manual
   running_the_federation.docker
   ../../plan_settings
   
