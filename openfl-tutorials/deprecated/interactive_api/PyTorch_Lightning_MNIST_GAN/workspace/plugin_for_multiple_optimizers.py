# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pytorch Framework Adapter plugin for multiple optimizers."""


from openfl.plugins.frameworks_adapters.pytorch_adapter import _get_optimizer_state
from openfl.plugins.frameworks_adapters.pytorch_adapter import FrameworkAdapterPlugin
from openfl.plugins.frameworks_adapters.pytorch_adapter import to_cpu_numpy


class FrameworkAdapterPluginforMultipleOpt(FrameworkAdapterPlugin):
    """Framework adapter plugin class for multiple optimizers."""

    def __init__(self):
        """Initialize framework adapter."""
        super().__init__()

    @staticmethod
    def get_tensor_dict(model, optimizers=None):
        """
        Extract tensor dict from a model and a list of optimizers.

        Returns:
        dict {weight name: numpy ndarray}
        """
        state = to_cpu_numpy(model.state_dict())
        if optimizers is not None:
            for opt in optimizers:
                if isinstance(opt, dict):
                    opt_state = _get_optimizer_state(opt['optimizer'])
                else:
                    opt_state = _get_optimizer_state(opt)

                state = {**state, **opt_state}

        return state
