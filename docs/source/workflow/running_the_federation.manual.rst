.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_manual:

***************
Manual Approach
***************

The procedure below summarizes the steps to set up an aggregator-based workflow manually.

1. Install the |productName| package on all the nodes in the federation. See :ref:`install_package>` for details.

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


    

.. toctree::
   :maxdepth: 2
   :hidden:

   running_the_federation.baremetal
   running_the_federation.certificates
   running_the_federation.start_nodes
      
