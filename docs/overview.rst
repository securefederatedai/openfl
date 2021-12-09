.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

==========
Overview
==========

.. toctree::

   `definitions_and_conventions`_
   `definitions_and_conventions`_
     

.. note::

   This project is continually being developed and improved. Expect changes to this manual, the project code, and the project design.
   
Open Federated Learning (OpenFL) is a Python\*\  3 project developed by Intel Internet of Things Group (IOTG) and Intel Labs.

.. figure:: images/ct_vs_fl.png

.. centered:: Federated Learning

.. _what_is_openfl:

***************************
What is Federated Learning?
***************************

`Federated learning <https://en.wikipedia.org/wiki/Federated_learning>`_ is a distributed machine learning approach that
enables collaboration on machine learning projects without sharing sensitive data, such as patient records, financial data,
or classified secrets (`McMahan, 2016 <https://arxiv.org/abs/1602.05629>`_;
`Sheller, Reina, Edwards, Martin, & Bakas, 2019 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6589345/>`_;
`Yang, Liu, Chen, & Tong, 2019 <https://arxiv.org/abs/1902.04885>`_; 
`Sheller et al., 2020 <https://www.nature.com/articles/s41598-020-69250-1>`_).
In federated learning, the model moves to meet the data rather than the data moving to meet the model. The movement of data across the federation are the model parameters and their updates.

.. figure:: images/diagram_fl_new.png

.. centered:: Federated Learning

.. _definitions_and_conventions:

***************************
Definitions and Conventions
***************************

Federated learning brings in a few more components to the traditional data science training pipeline:

Collaborator
	A collaborator is a client in the federation that has access to the local training, validation, and test datasets. By design, the collaborator is the only component of the federation with access to the local data. The local dataset should never leave the collaborator.
	
Aggregator
	A parameter server sends a global model to the collaborators. Parameter servers are often combined with aggregators on the same compute node.
	An aggregator receives locally tuned models from collaborators and combines the locally tuned models into a new global model. Typically, `federated averaging <https://arxiv.org/abs/1602.05629>`_, (a weighted average) is the algorithm used to combine the locally tuned models. 

Round
	A federation round is defined as the interval (typically defined in terms of training steps) where an aggregation is performed. Collaborators may perform local training on the model for multiple epochs (or even partial epochs) within a single training round.

.. toctree
..    overview.how_can_intel_protect_federated_learning
..    overview.what_is_intel_federated_learning
