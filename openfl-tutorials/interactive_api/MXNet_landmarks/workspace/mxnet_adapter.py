# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MXNet Framework Adapter plugin."""

from pickle import dumps
from pickle import loads
from typing import Dict

import mxnet as mx
import numpy as np
from mxnet import nd

from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface
)


class FrameworkAdapterPlugin(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    def __init__(self) -> None:
        """Initialize framework adapter."""

    @staticmethod
    def get_tensor_dict(model, optimizer=None) -> Dict[str, np.ndarray]:
        """
        Extract tensor dict from a model and an optimizer.

        Returns:
        dict {weight name: numpy ndarray}
        """
        state = {}
        if optimizer is not None:
            state = _get_optimizer_state(optimizer)

        model_params = model.collect_params()

        for param_name, param_tensor in model_params.items():
            if isinstance(param_tensor.data(), mx.ndarray.ndarray.NDArray):
                state[param_name] = param_tensor.list_data()[0].asnumpy()

        return state

    @staticmethod
    def set_tensor_dict(model, tensor_dict: Dict[str, np.ndarray],
                        optimizer=None, device=None) -> None:
        """
        Set tensor dict from a model and an optimizer.

        Given a dict {weight name: numpy ndarray} sets weights to
        the model and optimizer objects inplace.
        """
        if device is not None:
            device = mx.cpu() if device.startswith('cpu') else (
                mx.gpu(int(device.split(':')[1].strip()))
            )

        if optimizer is not None:
            _set_optimizer_state(optimizer, device, tensor_dict)
        model.collect_params().reset_ctx(device)

        model_params = model.collect_params()

        for param_name in model_params:
            model_params[param_name].set_data(nd.array(tensor_dict.pop(param_name), ctx=device))


def _get_optimizer_state(optimizer):
    """Return the optimizer state.

    Args:
        optimizer
    """
    states = loads(optimizer._updaters[0].get_states(dump_optimizer=False))
    result_states = {}
    for state_key, state_tuple in states.items():
        for state_ind, state in enumerate(state_tuple):
            result_states[f'opt_state__{state_key}__{state_ind}'] = state.asnumpy()

    return result_states


def _set_optimizer_state(optimizer, device, opt_state_dict):
    """Set the optimizer state.

    Args:
        optimizer:
        device:

    """
    state_keys, max_numstates = set(), 0
    for key in opt_state_dict.keys():
        if not key.startswith('opt_state'):
            continue
        _, part1, part2 = key.split('__')
        state_keys.add(int(part1))
        max_numstates = max(max_numstates, int(part2))

    out_state = {}
    for _ in range(len(state_keys)):
        key = state_keys.pop()
        state_vals = []
        for i in range(max_numstates + 1):
            state_vals.append(nd.array(opt_state_dict.pop(f'opt_state__{key}__{i}'), ctx=device))
        out_state[key] = tuple(state_vals)

    optimizer._updaters[0].set_states(dumps(out_state))
