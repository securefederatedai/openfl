.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_aggregato_based:

**********************
Running the Federation
**********************

First make sure you've installed the software :ref:`using these instructions <install_initial_steps>`

.. figure:: images/openfl_flow.png

.. centered:: 100K foot view of OpenFL workflow
    
The high-level workflow is shown in the figure above. Note that once OpenFL is installed on all nodes of the federation and every member of the federation has a valid PKI certificate, all that is needed to run an instance of a federated workload is to distribute the workspace to all federation members and then run the command to start the node (e.g. :code:`fx aggregator start`/:code:`fx collaborator start`). In other words, most of the work is setting up an initial environment on all of the federation nodes that can be used across new instantiations of federations.

.. toctree::
   :maxdepth: 4

   running_the_federation.agg_based.notebook
   running_the_federation.agg_based.baremetal
   running_the_federation.agg_based.docker
   running_the_federation.agg_based.certificates
   running_the_federation.agg_based.start_nodes.rst
   running_the_federation.director_based.interactive_api
