.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0.


.. Documentation master file, created by
   sphinx-quickstart on Thu Oct 24 15:07:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***************************************
Welcome to the |productName| documentation!
***************************************

Intel\ :sup:`®` \ Federated Learning (|productName|) is a Python3 library for federated learning.
It enables organizations to collaborately train a
model without sharing sensitive information with each other.

There are basically two components in the library:
the *collaborator* which uses local dataset to train
a global model and the *aggregator* which receives
model updates from collaborators and combines them to form 
the global model.

The *aggregator* is framework-agnostic, while the *collaborator*
can use any deep learning frameworks, such as `Tensorflow <https://www.tensorflow.org/>`_ or
`PyTorch <https://pytorch.org/>`_.

|productName| is developed by Intel Labs and Intel Internet of Things Group.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   manual
   openfl
   models
   data
   troubleshooting


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
