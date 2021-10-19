.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

******
|productName| Plugin Components
******

.. toctree::
   :maxdepth: 2

   framework_adapter_
   serializer_plugin_
   device_monitor_plugin_


|productName| is designed to be a flexible and extensible framework. Plugins are interchangeable parts of 
|productName| components. 
A plugin may be :code:`required` or :code:`optional`. |productName| can run without optional plugins. 
|productName| users are free to provide 
their implementations of |productName| plugins to achieve a desired behavior. 
Technically, a plugin is just a class, that satisfies a certain interface. One may enable a plugin by putting its 
import path and initialization parameters to the config file of a corresponding |productName| component 
or to the frontend Python API. Please refer to openfl-tutorials for more information.

.. _framework_adapter:

Framework Adapter
######################

Framework Adapter plugins enable |productName| support for Deep Learning frameworks usage in FL experiments. 
It is a required plugin for the frontend API component and Envoy.
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
It is a required plugin.
A Serializer plugin must implement :code:`serialize` method that creates a python object's representation on disk.

.. code-block:: python

   @staticmethod
   def serialize(object_, filename: str) -> None:

As well as :code:`restore_object` that will load previously serialized object from disk.

.. code-block:: python

   @staticmethod
   def restore_object(filename: str):


.. _device_monitor_plugin:

CUDA Device Monitor
######################

CUDA Device Monitor plugin is an optional plugin for Envoy that can gather status information about GPU devices. 
This information may be used by Envoy and included in a healthcheck message that is sent to Director. 
Thus the CUDA devices statuses are visible to frontend users that may query this Envoy Registry information from Director.

CUDA Device Monitor plugin must implement the following interface:

.. code-block:: python

   class CUDADeviceMonitor:
   
      def get_driver_version(self) -> str:
         ...

      def get_device_memory_total(self, index: int) -> int:
         ...

      def get_device_memory_utilized(self, index: int) -> int:
         ...

      def get_device_utilization(self, index: int) -> str:
         """It is just a general method that returns a string that may be shown to the frontend user."""
         ...


