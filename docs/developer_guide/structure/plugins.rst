.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

*****************
Plugin Components
*****************

Open Federated Learning (OpenFL) is designed to be a flexible and extensible framework. Plugins are interchangeable parts of OpenFL components. Different plugins support varying usage scenarios.
A plugin may be **required** or **optional**. 

You can provide your implementations of OpenFL plugins to achieve a desired behavior. Technically, a plugin is just a class that implements some interface. You may enable a plugin by putting its 
import path and initialization parameters to the config file of a corresponding OpenFL component or to the frontend Python API. See `openfl-tutorials <https://github.com/intel/openfl/tree/develop/openfl-tutorials>`_ for more details.

.. _framework_adapter:

Framework Adapter
######################

The Framework Adapter plugin enables OpenFL support for Deep Learning frameworks usage in FL experiments. 
It is a **required** plugin for the frontend API component and Envoy.
All the framework-specific operations on model weights are isolated in this plugin so OpenFL can be framework-agnostic.

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