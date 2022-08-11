# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom model DeviceArray - JAX Numpy adapter."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .framework_adapter_interface import FrameworkAdapterPluginInterface

_DELIM = '.'

class FrameworkAdapterPlugin(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    @staticmethod
    def get_tensor_dict(model, optimizer=None, suffix=''):
        """
        Extract tensor dict from a model.params and model.opt_state (optimizer).

        Returns:
        dict {weight name: numpy ndarray}

        """

        # Convert PyTree Structure DeviceArray to Numpy
        model_params = jax.tree_util.tree_map(np.array, model.params)
        params_dict = _get_weights_dict(model_params, 'param', suffix)
        
        if isinstance(model.opt_state[0], optax.TraceState):
            model_opt_state = jax.tree_util.tree_map(np.array, model.opt_state)[0][0]
            opt_dict = _get_weights_dict(model_opt_state, 'opt', suffix)
            params_dict.update(opt_dict)

        return params_dict

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        """
        Set the `model.params and model.opt_state` with a flattened tensor dictionary.
        Choice of JAX platform (device) cpu/gpu/gpu is initialized at start.
        Args:
            tensor_dict: flattened {weight name: numpy ndarray} tensor dictionary

        Returns:
            None
        """
        
        tensor_dict = jax.tree_util.tree_map(jnp.array, tensor_dict)

        _set_weights_dict(model, tensor_dict, 'param')

        if isinstance(model.opt_state[0], optax.TraceState):
            _set_weights_dict(model, tensor_dict, 'opt')            


def _set_weights_dict(obj, weights_dict, prefix=''):
    """Set the object weights with a dictionary.

    The obj can be a model or an optimizer.
    
    Args:
        obj (Model or Optimizer): The target object that we want to set
        the weights.
        weights_dict (dict): The weight dictionary.

    Returns:
        None
    """
    
    model_state_dict = obj.opt_state[0][0] if prefix == 'opt' else obj.params       

    for layer_name, param_obj in model_state_dict.items():
        for param_name, value in param_obj.items():
            key = _DELIM.join(filter(None, [prefix, layer_name, param_name]))
            if key in weights_dict:
                model_state_dict[layer_name][param_name] = weights_dict[key]


def _get_weights_dict(obj, prefix='', suffix=''):
    """
    Get the dictionary of weights.

    Parameters
    ----------
    obj : Model or Optimizer
        The target object that we want to get the weights.

    Returns
    -------
    dict
        The weight dictionary.
    """
    weights_dict = dict()
    for layer_name, param_obj in obj.items():
        for param_name, value in param_obj.items():
            key = _DELIM.join(filter(None, [prefix, layer_name, param_name, suffix]))
            weights_dict[key] = value

    return weights_dict