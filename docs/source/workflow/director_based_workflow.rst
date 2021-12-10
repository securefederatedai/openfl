.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _director_workflow:

************************
Director-Based Workflow
************************

A director-based workflow uses long-lived components in a federation. These components continue to be available to distribute more experiments in the federation.
	
- The *Director* is the central node of the federation. This component starts an *Aggregator* for each experiment, sends data to connected collaborator nodes, and provides updates on the status.
- The *Envoy* runs on collaborator nodes connected to the *Director*. When the *Director* starts an experiment, the *Envoy* starts the *Collaborator* to train the global model.


The director-based workflow comprises the following roles and their tasks:

    - :ref:`establishing_federation_director`
    - :ref:`establishing_federation_envoy`
    - :ref:`establishing_federation_experiment_manager`

Follow the procedure in the director-based workflow to become familiar with the setup required and APIs provided for each role in the federation: *Experiment manager (Data scientist)*, *Director manager*, and *Collaborator manager*. 

- *Experiment manager* (or Data scientist) is a person or group of people using OpenFL.
- *Director Manager* is ML model creator's representative controlling Director.
- *Collaborator manager* is Data onwer's representative controlling Envoy.


.. note::
    The Open Federated Learning (|productName|) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter\*\  notebook or a Python\*\  script. 
    
    See :doc:`director_based_workflow.interactive_api` for details.

An overview of this workflow is shown below.

.. figure:: ../openfl/director_workflow.svg

.. centered:: Overview of the Director-Based Workflow


.. toctree::
   :maxdepth: 2
   :hidden:

   director_based_workflow.roles
   director_based_workflow.interactive_api

