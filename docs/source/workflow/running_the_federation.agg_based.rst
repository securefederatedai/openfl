.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_aggregator_based:

*************************
Aggregator-Based Workflow
*************************

This workflow uses short-lived components in a federation, which is terminated when the experiment is finished. The components are as follows:

- The *Collaborator* uses a local dataset to train a global model and the *Aggregator* receives model updates from collaborators and combines them to form the global model.
- The *Aggregator* is framework-agnostic, while the *Collaborator* can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_\* \  or `PyTorch <https://pytorch.org/>`_\*\.

An overview of this workflow is shown below.

.. figure:: /images/openfl_flow.png

.. centered:: Overview of the Aggregator-Based Workflow



There are two ways to run this workflow:

- Docker approach
- Manual approach



.. toctree::
   :maxdepth: 2

   source/workflow/running_the_federation.docker
   source/workflow/running_the_federation.manual
   
