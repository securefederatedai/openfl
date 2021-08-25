.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation:

**********************
Running the Federation
**********************

First make sure you've installed the software :ref:`using these instructions <install_initial_steps>`

.. figure:: images/openfl_flow.png

.. centered:: 100K foot view of OpenFL workflow
    
The high-level workflow is shown in the figure above. Note that once OpenFL is installed on all nodes of the federation and every member of the federation has a valid PKI certificate, all that is needed to run an instance of a federated workload is to distribute the workspace to all federation members and then run the command to start the node (e.g. :code:`fx aggregator start`/:code:`fx collaborator start`). In other words, most of the work is setting up an initial environment on all of the federation nodes that can be used across new instantiations of federations.

.. toctree::
   :maxdepth: 2

   source/workflow/running_the_federation.agg_based
   source/workflow/director_based_workflow
