# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Custom model DeviceArray - JAX Numpy adapter."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util

from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface,
)


class FrameworkAdapterPlugin(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    @staticmethod
    def get_tensor_dict(model, optimizer=None):
        """Extract tensor dict from a model.params and model.opt_state
        (optimizer).

        Args:
            model (object): The model object.
            optimizer (object, optional): The optimizer object. Defaults to None.

        Returns:
            params_dict (dict): A dictionary with weight name as key and numpy ndarray as value.
        """

        # Convert PyTree Structure DeviceArray to Numpy
        model_params = jax.tree_util.tree_map(np.array, model.params)
        params_dict = _get_weights_dict(model_params, "param")

        # If optimizer is initialized
        # Optax Optimizer agnostic state processing (TraceState,
        # AdamScaleState, any...)
        if not isinstance(model.opt_state[0], optax.EmptyState):
            opt_state = jax.tree_util.tree_map(np.array, model.opt_state)[0]
            opt_vars = filter(_get_opt_vars, dir(opt_state))
            for var in opt_vars:
                opt_dict = getattr(opt_state, var)  # Returns a dict
                # Flattens a deeply nested dictionary
                opt_dict = _get_weights_dict(opt_dict, f"opt_{var}")
                params_dict.update(opt_dict)

        return params_dict

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device="cpu"):
        """
        Set the `model.params and model.opt_state` with a flattened tensor
        dictionary. Choice of JAX platform (device) cpu/gpu/gpu is initialized
        at start.
        Args:
            model (object): The model object.
            tensor_dict (dict): Flattened dictionary with weight name as key
                and numpy ndarray as value.
            optimizer (object, optional): The optimizer object. Defaults to
                None.
            device (str, optional): The device to be used. Defaults to 'cpu'.

        Returns:
            None
        """

        tensor_dict = jax.tree_util.tree_map(jnp.array, tensor_dict)

        _set_weights_dict(model, tensor_dict, "param")

        if not isinstance(model.opt_state[0], optax.EmptyState):
            _set_weights_dict(model, tensor_dict, "opt")


def _get_opt_vars(x):
    """Helper function to filter out unwanted variables.

    Args:
        x (str): The variable name.

    Returns:
        bool: True if the variable is wanted, False otherwise.
    """
    return False if x.startswith("_") or x in ["index", "count"] else True


def _set_weights_dict(obj, weights_dict, prefix=""):
    """Set the object weights with a dictionary.

    The obj can be a model or an optimizer.

    Args:
        obj (Model or Optimizer): The target object that we want to set the
            weights.
        weights_dict (dict): The weight dictionary.
        prefix (str, optional): The prefix for the weight dictionary keys.
            Defaults to ''.

    Returns:
        None
    """

    if prefix == "opt":
        model_state_dict = obj.opt_state[0]
        # opt_vars -> ['mu', 'nu'] for Adam or ['trace'] for SGD or ['ANY'] for any
        opt_vars = filter(_get_opt_vars, dir(model_state_dict))
        for var in opt_vars:
            opt_state_dict = getattr(model_state_dict, var)
            _update_weights(opt_state_dict, weights_dict, prefix, var)
    else:
        _update_weights(obj.params, weights_dict, prefix)


def _update_weights(state_dict, tensor_dict, prefix, suffix=None):
    """Helper function to update the weights of the state dictionary.

    Re-assignment of the state variable(s) is restricted.
    Instead update the nested layers weights iteratively.

    Args:
        state_dict (dict): The state dictionary.
        tensor_dict (dict): The tensor dictionary.
        prefix (str): The prefix for the weight dictionary keys.
        suffix (str, optional): The suffix for the weight dictionary keys.
            Defaults to None.

    Returns:
        None
    """
    dict_prefix = f"{prefix}_{suffix}" if suffix is not None else f"{prefix}"
    for layer_name, param_obj in state_dict.items():
        for param_name, value in param_obj.items():
            key = "*".join([dict_prefix, layer_name, param_name])
            if key in tensor_dict:
                state_dict[layer_name][param_name] = tensor_dict[key]


def _get_weights_dict(obj, prefix):
    """Get the dictionary of weights.

    Args:
        obj (Model or Optimizer): The target object that we want to get the
            weights.
        prefix (str): The prefix for the weight dictionary keys.

    Returns:
        flat_params (dict): The weight dictionary.
    """
    weights_dict = {prefix: obj}
    # Flatten the dictionary with a given separator for
    # easy lookup and assignment in `set_tensor_dict` method.
    flat_params = traverse_util.flatten_dict(weights_dict, sep="*")
    return flat_params
