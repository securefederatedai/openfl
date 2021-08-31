.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

******
|productName| Plugin Components
******

.. toctree::
   :maxdepth: 2

   framework_adapter_
   serializer_plugin_


|productName| is designed to be a flexible and extensible framework. Plugins are interchangeable parts of 
|productName| components. Different plugins support varying usage scenarios. |productName| users are free to provide 
their implementations of |productName| plugins to support desired behavior.

.. _framework_adapter:

Framework Adapter
######################

Framework Adapter plugins enable |productName| support for Deep Learning frameworks usage in FL experiments. 
All the framework-specific operations on model weights are isolated in this plugin so |productName| can be framework-agnostic.
The Framework adapter plugin interface is simple: there are two required methods to load and extract tensors from 
a model and an optimizer. 

:code:`get_tensor_dict` method accepts a model and optionally an optimizer. It should return a dictionary :code:`{tensor_name : ndarray}` 
that maps tensor names to tensors in the NumPy representation.

.. code-block:: python

   @staticmethod
   def get_tensor_dict(model, optimizer=None) -> dict:

:code:`set_tensor_dict` method accepts a tensor dictionary, a model, and optionally an optimizer. It loads weights from the tensor dictionary 
to the model in place. Tensor names in the dictionary match corresponding names set in :code:`get_tensor_dict`

.. code-block:: python

   @staticmethod
   def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu') -> None:

If your new framework model cannot be directly serialized with pickle-type libraries, you can optionally 
implement the :code:`serialization_setup` method to prepare the model object for serialization.

.. code-block:: python

   def serialization_setup(): 


.. _serializer_plugin:

Experiment Serializer
######################

Serializer plugins are used on the Frontend API to serialize the Experiment components and then on Envoys to deserialize them back.
Currently, the default serializer is based on pickling.

A Serializer plugin must implement :code:`serialize` method that creates a python object's representation on disk.

.. code-block:: python

   @staticmethod
   def serialize(object_, filename: str) -> None:

As well as :code:`restore_object` that will load previously serialized object from disk.

.. code-block:: python

   @staticmethod
   def restore_object(filename: str):
