.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation:

**********************
Running the Federation
**********************

You can use the following workflows to create a learning federation.

.. _running_the_federation_aggregator_workflow:

Aggregator-Based Workflow
=========================
The :ref:`aggregator-based workflow <running_the_federation_aggregator_based>` uses short-lived components in a federation, which is terminated when the experiment is finished. 

- The *Collaborator* uses a local dataset to train a global model and the *Aggregator* receives model updates from collaborators and combines them to form the global model.
- The *Aggregator* is framework-agnostic, while the *Collaborator* can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_ or `PyTorch <https://pytorch.org/>`_.

.. note::

	Follow the procedure in the aggregator-based workflow to become familiar with the APIs in |productName| and conventions such as FL Plans, aggregators, and collaborators. 


.. _running_the_federation_director_workflow:

Director-Based Workflow
=======================
The :ref:`director-based workflow <director_based_workflow>` uses long-lived components in a federation. These components continue to be available to distribute more experiments in the federation.
	
- The *Director* is the central node of the federation. This component starts an *Aggregator* for each experiment, sends data to connected collaborator nodes, and provides updates on the status.
- The *Envoy* runs on collaborator nodes connected to the *Director*. When the *Director* starts an experiment, the *Envoy* starts the *Collaborator* to train the global model.
	
.. note::

	Follow the procedure in the director-based workflow to become familiar with the setup required and APIs provided for each role in the federation: Director manager, Envoy manager, and Experiment manager (data scientist). 


.. toctree::
   :maxdepth: 2
   :hidden:

   source/workflow/running_the_federation.agg_based
   source/workflow/director_based_workflow
