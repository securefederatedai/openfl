# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Pytorch Framework Adapter plugin."""
from copy import deepcopy

import numpy as np
import torch as pt

from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface,
)


class FrameworkAdapterPlugin(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    def __init__(self) -> None:
        """Initialize framework adapter."""
        pass

    @staticmethod
    def get_tensor_dict(model, optimizer=None):
        """Extract tensor dict from a model and an optimizer.

        Args:
            model (object): The model object.
            optimizer (object, optional): The optimizer object. Defaults to
                None.

        Returns:
            dict: A dictionary with weight name as key and numpy ndarray as
                value.
        """
        state = to_cpu_numpy(model.state_dict())

        if optimizer is not None:
            opt_state = _get_optimizer_state(optimizer)
            state = {**state, **opt_state}

        return state

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device="cpu"):
        """
        Set tensor dict from a model and an optimizer.

        Given a dict {weight name: numpy ndarray} sets weights to
        the model and optimizer objects inplace.

        Args:
            model (object): The model object.
            tensor_dict (dict): The tensor dictionary.
            optimizer (object, optional): The optimizer object. Defaults to
                None.
            device (str, optional): The device to be used. Defaults to 'cpu'.

        Returns:
            None
        """
        new_state = {}
        # Grabbing keys from model's state_dict helps to confirm we have
        # everything
        for k in model.state_dict():
            new_state[k] = pt.from_numpy(tensor_dict.pop(k)).to(device)

        # set model state
        model.load_state_dict(new_state)

        if optimizer is not None:
            # see if there is state to restore first
            if tensor_dict.pop("__opt_state_needed") == "true":
                _set_optimizer_state(optimizer, device, tensor_dict)

            # sanity check that we did not record any state that was not used
            assert len(tensor_dict) == 0


def _set_optimizer_state(optimizer, device, derived_opt_state_dict):
    """Set the optimizer state.

    Args:
        optimizer (object): The optimizer object.
        device (str): The device to be used.
        derived_opt_state_dict (dict): The derived optimizer state dictionary.

    Returns:
        None
    """
    temp_state_dict = expand_derived_opt_state_dict(derived_opt_state_dict, device)

    # Setting other items from the param_groups
    # getting them from the local optimizer
    # (expand_derived_opt_state_dict sets only 'params')
    for i, group in enumerate(optimizer.param_groups):
        for k, v in group.items():
            if k not in temp_state_dict["param_groups"][i]:
                temp_state_dict["param_groups"][i][k] = v

    optimizer.load_state_dict(temp_state_dict)


def _get_optimizer_state(optimizer):
    """Return the optimizer state.

    Args:
        optimizer (object): The optimizer object.

    Returns:
        derived_opt_state_dict (dict): The optimizer state dictionary.
    """
    opt_state_dict = deepcopy(optimizer.state_dict())

    # Optimizer state might not have some parts representing frozen parameters
    # So we do not synchronize them
    param_keys_with_state = set(opt_state_dict["state"].keys())
    for group in opt_state_dict["param_groups"]:
        local_param_set = set(group["params"])
        params_to_sync = local_param_set & param_keys_with_state
        group["params"] = sorted(params_to_sync)

    derived_opt_state_dict = _derive_opt_state_dict(opt_state_dict)

    return derived_opt_state_dict


def _derive_opt_state_dict(opt_state_dict):
    """Separate optimizer tensors from the tensor dictionary.

    Flattens the optimizer state dict so as to have key, value pairs with
    values as numpy arrays.
    The keys have sufficient info to restore opt_state_dict using
    expand_derived_opt_state_dict.

    Args:
        opt_state_dict (dict): The optimizer state dictionary.

    Returns:
        derived_opt_state_dict (dict): The derived optimizer state dictionary.
    """
    derived_opt_state_dict = {}

    # Determine if state is needed for this optimizer.
    if len(opt_state_dict["state"]) == 0:
        derived_opt_state_dict["__opt_state_needed"] = "false"
        return derived_opt_state_dict

    derived_opt_state_dict["__opt_state_needed"] = "true"

    # Using one example state key, we collect keys for the corresponding
    # dictionary value.
    example_state_key = opt_state_dict["param_groups"][0]["params"][0]
    example_state_subkeys = set(opt_state_dict["state"][example_state_key].keys())

    # We assume that the state collected for all params in all param groups is
    # the same.
    # We also assume that whether or not the associated values to these state
    # subkeys is a tensor depends only on the subkey.
    # Using assert statements to break the routine if these assumptions are
    # incorrect.
    for state_key in opt_state_dict["state"].keys():
        assert example_state_subkeys == set(opt_state_dict["state"][state_key].keys())
        for state_subkey in example_state_subkeys:
            assert isinstance(
                opt_state_dict["state"][example_state_key][state_subkey],
                pt.Tensor,
            ) == isinstance(opt_state_dict["state"][state_key][state_subkey], pt.Tensor)

    state_subkeys = list(opt_state_dict["state"][example_state_key].keys())

    # Tags will record whether the value associated to the subkey is a
    # tensor or not.
    state_subkey_tags = []
    for state_subkey in state_subkeys:
        if isinstance(opt_state_dict["state"][example_state_key][state_subkey], pt.Tensor):
            state_subkey_tags.append("istensor")
        else:
            state_subkey_tags.append("")
    state_subkeys_and_tags = list(zip(state_subkeys, state_subkey_tags))

    # Forming the flattened dict, using a concatenation of group index,
    # subindex, tag, and subkey inserted into the flattened dict key -
    # needed for reconstruction.
    nb_params_per_group = []
    for group_idx, group in enumerate(opt_state_dict["param_groups"]):
        for idx, param_id in enumerate(group["params"]):
            for subkey, tag in state_subkeys_and_tags:
                if tag == "istensor":
                    new_v = opt_state_dict["state"][param_id][subkey].cpu().numpy()
                else:
                    new_v = np.array([opt_state_dict["state"][param_id][subkey]])
                derived_opt_state_dict[f"__opt_state_{group_idx}_{idx}_{tag}_{subkey}"] = new_v
        nb_params_per_group.append(idx + 1)
    # group lengths are also helpful for reconstructing
    # original opt_state_dict structure
    derived_opt_state_dict["__opt_group_lengths"] = np.array(nb_params_per_group)

    return derived_opt_state_dict


def expand_derived_opt_state_dict(derived_opt_state_dict, device):
    """Expand the optimizer state dictionary.

    Takes a derived opt_state_dict and creates an opt_state_dict suitable as
    input for load_state_dict for restoring optimizer state.

    Reconstructing state_subkeys_and_tags using the example key
    prefix, "__opt_state_0_0_", certain to be present.

    Args:
        derived_opt_state_dict (dict): The derived optimizer state dictionary.
        device (str): The device to be used.

    Returns:
        opt_state_dict (dict): The expanded optimizer state dictionary.
    """
    state_subkeys_and_tags = []
    for key in derived_opt_state_dict:
        if key.startswith("__opt_state_0_0_"):
            stripped_key = key[16:]
            if stripped_key.startswith("istensor_"):
                this_tag = "istensor"
                subkey = stripped_key[9:]
            else:
                this_tag = ""
                subkey = stripped_key[1:]
            state_subkeys_and_tags.append((subkey, this_tag))

    opt_state_dict = {"param_groups": [], "state": {}}
    nb_params_per_group = list(derived_opt_state_dict.pop("__opt_group_lengths").astype(np.int32))

    # Construct the expanded dict.
    for group_idx, nb_params in enumerate(nb_params_per_group):
        these_group_ids = [f"{group_idx}_{idx}" for idx in range(nb_params)]
        opt_state_dict["param_groups"].append({"params": these_group_ids})
        for this_id in these_group_ids:
            opt_state_dict["state"][this_id] = {}
            for subkey, tag in state_subkeys_and_tags:
                flat_key = f"__opt_state_{this_id}_{tag}_{subkey}"
                if tag == "istensor":
                    new_v = pt.from_numpy(derived_opt_state_dict.pop(flat_key))
                else:
                    # Here (for currrently supported optimizers) the subkey
                    # should be 'step' and the length of array should be one.
                    assert subkey == "step"
                    assert len(derived_opt_state_dict[flat_key]) == 1
                    new_v = int(derived_opt_state_dict.pop(flat_key))
                opt_state_dict["state"][this_id][subkey] = new_v

    # sanity check that we did not miss any optimizer state
    assert len(derived_opt_state_dict) == 0, str(derived_opt_state_dict)

    return opt_state_dict


def to_cpu_numpy(state):
    """Send data to CPU as Numpy array.

    Args:
        state (dict): The state dictionary.

    Returns:
        state (dict): The state dictionary with all values as numpy arrays.
    """
    # deep copy so as to decouple from active model
    state = deepcopy(state)

    for k, v in state.items():
        # When restoring, we currently assume all values are tensors.
        if not pt.is_tensor(v):
            raise ValueError(
                "We do not currently support non-tensors " "coming from model.state_dict()"
            )
        # get as a numpy array, making sure is on cpu
        state[k] = v.cpu().numpy()
    return state
