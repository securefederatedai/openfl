# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base abstract optimizer class module."""
import abc
from importlib import import_module
from os.path import splitext
from typing import Dict

from numpy import ndarray

from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface
)


class Optimizer(abc.ABC):
    """Base abstract optimizer class."""

    @abc.abstractmethod
    def step(self, gradients: Dict[str, ndarray]) -> None:
        """Perform a single step for parameter update.

        Args:
            gradients: Partial derivatives with respect to optimized parameters.
        """
        pass

    def _set_params_from_model(self, model_interface):
        """Eject and store model parameters."""
        class_name = splitext(model_interface.framework_plugin)[1].strip('.')
        module_path = splitext(model_interface.framework_plugin)[0]
        framework_adapter = import_module(module_path)
        framework_adapter_plugin: FrameworkAdapterPluginInterface = getattr(
            framework_adapter, class_name, None)
        self.params: Dict[str, ndarray] = framework_adapter_plugin.get_tensor_dict(
            model_interface.provide_model())
