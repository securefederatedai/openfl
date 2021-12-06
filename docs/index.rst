.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0.


.. Documentation master file, created by
   sphinx-quickstart on Thu Oct 24 15:07:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************************************************
Welcome to the Open Federated Learning (|productName|) Documentation!
*********************************************************************

Open Federated Learning (|productName|) is a Python\* \ 3 library for federated learning that enables organizations to collaboratively train a model without sharing sensitive information.

The aggregation to create the global training model is framework-agnostic, while the collaboration from local training models using local datasets supports any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_\* \ or `PyTorch <https://pytorch.org/>`_\*\, via a plugin framework.

.. |productName| has the following components:

.. Short-Lived Components
..	These components are terminated when the experiment is finished.
	
..	- The *Collaborator* which uses local dataset to train a global model and the *Aggregator* which receives model updates from collaborators and combines them to form the global model.
..	- The *Aggregator* is framework-agnostic, while the *Collaborator* can use any deep learning frameworks, such as `TensorFlow <https://www.tensorflow.org/>`_ or `PyTorch <https://pytorch.org/>`_.
	

.. Long-Lived Components
..	These components continue to be available to distribute more experiments in the federation.
	
..	- The *Director* is the central node of the federation. This component starts an *Aggregator* for each experiment, sends data to connected collaborator nodes, and provides updates on the status.
..	- The *Envoy* runs on collaborator nodes connected to the *Director*. When the *Director* starts an experiment, the *Envoy* starts the *Collaborator* to train the global model.
	
.. For details, see :ref:`openfl_components`.

|productName| is developed by Intel Labs and Intel Internet of Things Group.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   manual
   openfl
   troubleshooting
   notices_and_disclaimers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
