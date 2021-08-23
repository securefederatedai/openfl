.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

******
|productName| Plugin Components
******

.. toctree::
   :maxdepth: 2

   framework_adapter_
   serializer_plugin_

.. _framework_adapter:

Framework Adapter
######################

Framework Adapter plugins enable |productName| support for Deep Learning frameworks usage in FL experiments. 
All the framework-specific operations on model weights are isolated in this plugin so |productName| can be framework-agnostic.
The Framework adapter plugin interface is simple: there are two required methods to load and extract tensors from 
a model and an optimizer. 

:code:`get_tensor_dict` method accepts a model and optionally an optimizer. It should return a dictionary :code:`{tensor_name : ndarray}` 
that maps tensor names to tensors in the numpy representation.

.. code-block:: python

   @staticmethod
   def get_tensor_dict(model, optimizer=None) -> dict:

:code:`set_tensor_dict` method accepts a tensor dictionary, a model and optionally an optimizer. It loads weights from tensor dictionary 
to the model inplace. Tensor names in the dictionary matches corresponding names set in :code:`get_tensor_dict`

.. code-block:: python

   @staticmethod
   def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu') -> None:

Implement :code:`serialization_setup` optional method if some preparation are required before the model serialization. 
This method would be called on the frontend Python API during an FL experiment extraction to the Director side.

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

As well as :code:`restore_object` that will load previously serialized object from disc.

.. code-block:: python

   @staticmethod
   def restore_object(filename: str):
