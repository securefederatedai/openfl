.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_manual:

*******************
Bare Metal Approach
*******************

.. note::

    Ensure you have installed the |productName| package on every node (aggregator and collaborators) in the federation.

    See :ref:`install_package` for details.


You can use the `"Hello Federation" bash script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.sh>`_ to quickly create a federation (an aggregator node and two collaborator nodes) to test the project pipeline.

.. literalinclude:: ../tests/github/test_hello_federation.sh
  :language: bash

However, continue with the following procedure for details in creating a federation with an aggregator-based workflow.

    :doc:`STEP 1: Create a Workspace on the Aggregator <running_the_federation.baremetal>`

        - Creates a federated learning workspace on one of the nodes.


    :doc:`STEP 2: Configure the Federation <running_the_federation.certificates>`

        - Ensures each node in the federation has a valid public key infrastructure (PKI) certificate.
        - Distributes the workspace from the aggregator node to the other collaborator nodes.


    :doc:`STEP 3: Start the Federation <running_the_federation.start_nodes>`


    

.. toctree::
   :maxdepth: 3
   :hidden:

   running_the_federation.baremetal
   running_the_federation.certificates
   running_the_federation.start_nodes
      
