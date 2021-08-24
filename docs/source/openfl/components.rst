.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _openfl_components:

******
|productName| core components
******

.. toctree::
   :maxdepth: 2

   `Spawning`_
   `Long-living`_


.. _openfl_spawning_components:

Spawning components
##########

Aggregator
===========

The aggregator is a short-living entity, which means that its lifespan is limited by experiment execution time. 
It orchestrates collaborators according to the FL plan and performs model updates aggregation.
The aggregator is spawned by the Director (described below) when a new experiment is submitted.


Collaborator
=============

Collaborator is also a short living entity, it manages training the model on local data: executes assigned tasks, 
converts DL framework-specific tensor objects to |productName| inner representation, and exchanges model parameters with the aggregator.
Converting tensors is done by :ref:`Framework adapter <framework_adapter>` plugins. |productName| ships with Pytorch and Tensorflow 2.x framework adapters. 
These framework adapters are intended to be extensible, 
and we encourage users to contribute new adapters for DL frameworks they would like to see supported in |productName|. 

Model is loaded with relevant weights before every task and at the end of the training task, weights are extracted to be sent to the central node and aggregated.
Collaborator instance is created by Envoy (described below) when a new experiment is submitted. 
Every collaborator is a unique service as it is loaded with a local Shard Descriptor to perform tasks included in an FL experiment.

.. _openfl_ll_components:

Long-living components
#############

Director
==========

Director is a long-living entity; it is a central node of the federation and may take in several experiments (with the same data interface). When an experiment is reported director starts an aggregator and sends the experiment data to involved envoys; during the experiment, Director oversees the aggregator and updates the user on the status of the experiment.
Director runs two services: one for frontend users and another one for envoys. It can distribute an experiment reported with the frontend API across the federation and communicate back a trained model snapshot and metrics.
Director support several concurrent frontend connections (yet experiments are run one by one)

Envoy
=========

|productName| comes with another long-existing actor called Envoy. It runs on collaborator machines connected to a *Director*. 
There is one to one mapping between *Envoys* and Dataset shards: every *Envoy* needs exactly one 
*`Shard Descriptor <https://github.com/intel/openfl/blob/develop/openfl/interface/interactive_api/shard_descriptor.py>`_* to run. 
When the *Director* starts an experiment, *Envoy* will accept the experiment workspace, prepare the environment and start a *Collaborator*.
