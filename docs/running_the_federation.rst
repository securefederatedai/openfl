.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation:

******************
Run the Federation
******************

OpenFL currently supports two types of workflow for how to set up and run a federation: Director-based workflow (preferrable) and Aggregator-based workflow (old workflow, will not be supported soon). Director-based workflow introduces a new and more convenient way to set up a federation and brings "long-lived" components in a federation ("Director" and "Envoy").

:doc:`source/workflow/director_based_workflow`
    A federation created with this workflow continues to be available to distribute more experiments in series.

:doc:`source/workflow/running_the_federation.agg_based`
    With this workflow, the federation is terminated when the experiment is finished.   


.. toctree::
   :maxdepth: 2
   :hidden:

   source/workflow/running_the_federation.agg_based
   source/workflow/director_based_workflow


   
