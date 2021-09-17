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


The procedure below summarizes the steps to set up an aggregator-based workflow.

1. Install the |productName| package on all the nodes in the federation :ref:`using these instructions <install_package>`.

2. Create a federated learning workspace on one of the nodes. This node is called the aggregator.

3. Distribute the workspace from the aggregator node to the other collaborator nodes.

4. Ensure each node in the federation has a valid PKI certificate.

5. Start the federation.

	- On the aggregator node.

		.. code-block:: console

			fx aggregator start

		
	- On the each of the collaborator nodes.

		.. code-block:: console

			fx collaborator start


There are two ways to run this workflow:

- :ref:`Docker approach </source/workflow/running_the_federation.docker>`
- :ref:`Manual approach </source/workflow/running_the_federation.baremetal>`



.. toctree::
   :maxdepth: 2
   :hidden:

   running_the_federation.docker
   running_the_federation.baremetal
   running_the_federation.certificates
   running_the_federation.start_nodes.rst

