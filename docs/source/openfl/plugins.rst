.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

*****************
Plugin Components
*****************

.. toctree::
   :maxdepth: 2

   framework_adapter_
   serializer_plugin_
   device_monitor_plugin_

Open Federated Learning (|productName|) is designed to be a flexible and extensible framework. Plugins are interchangeable parts of |productName| components. Different plugins support varying usage scenarios.
A plugin may be **required** or **optional**. 

You can provide your implementations of |productName| plugins to achieve a desired behavior. Technically, a plugin is just a class that implements some interface. You may enable a plugin by putting its 
import path and initialization parameters to the config file of a corresponding |productName| component or to the frontend Python API. See `openfl-tutorials <https://github.com/intel/openfl/tree/develop/openfl-tutorials>`_ for more details.

.. _framework_adapter:

Framework Adapter
######################

The Framework Adapter plugin enables |productName| support for Deep Learning frameworks usage in FL experiments. 
It is a **required** plugin for the frontend API component and Envoy.
All the framework-specific operations on model weights are isolated in this plugin so |productName| can be framework-agnostic.

The Framework adapter plugin interface has two required methods to load and extract tensors from a model and an optimizer:

    - :code:`get_tensor_dict`
    - :code:`set_tensor_dict`

:code:`get_tensor_dict` method accepts a model and optionally an optimizer. It should return a dictionary :code:`{tensor_name : ndarray}` 
that maps tensor names to tensors in the NumPy representation.

    .. code-block:: python

       @staticmethod
       def get_tensor_dict(model, optimizer=None) -> dict:

:code:`set_tensor_dict` method accepts a tensor dictionary, a model, and optionally an optimizer. It loads weights from the tensor dictionary 
to the model in place. Tensor names in the dictionary match corresponding names set in :code:`get_tensor_dict`.

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

The Serializer plugin is used on the frontend Python API to serialize the Experiment components and then on Envoys to deserialize them.
Currently, the default serializer plugin is based on pickling. It is a **required** plugin.

The serializer plugin must implement the :code:`serialize` method that creates a Python object representation on disk.

    .. code-block:: python

       @staticmethod
       def serialize(object_, filename: str) -> None:

The plugin must also implement the :code:`restore_object` method that will load previously serialized object from disk.

    .. code-block:: python

       @staticmethod
       def restore_object(filename: str):


.. _device_monitor_plugin:

CUDA Device Monitor
######################

The CUDA Device Monitor plugin is an **optional** plugin for Envoys that can gather status information about GPU devices. 
This information may be used by Envoys and included in a healthcheck message that is sent to the Director. 
Therefore, you can query this Envoy Registry information from the Director to determine the status of CUDA devices.

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


