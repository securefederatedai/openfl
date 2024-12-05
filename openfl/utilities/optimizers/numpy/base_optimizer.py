# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Base abstract optimizer class module."""

import abc
from importlib import import_module
from os.path import splitext
from typing import Dict

from numpy import ndarray

from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface,
)


class Optimizer(abc.ABC):
    """Base abstract optimizer class.

    This class serves as a base class for all optimizers. It defines the basic
    structure that all derived optimizer classes should follow.
    It includes an abstract method `step` that must be implemented by any
    concrete optimizer class.
    """

    @abc.abstractmethod
    def step(self, gradients: Dict[str, ndarray]) -> None:
        """Perform a single step for parameter update.

        This method should be overridden by all subclasses to implement the
        specific optimization algorithm.

        Args:
            gradients (dict): Partial derivatives with respect to optimized
                parameters.
        """
        pass

    def _set_params_from_model(self, model_interface):
        """
        Eject and store model parameters.

        This method is used to extract the parameters from the provided model
        interface and store them in the optimizer.

        Args:
            model_interface: The model interface instance to provide
                parameters.
        """
        class_name = splitext(model_interface.framework_plugin)[1].strip(".")
        module_path = splitext(model_interface.framework_plugin)[0]
        framework_adapter = import_module(module_path)
        framework_adapter_plugin: FrameworkAdapterPluginInterface = getattr(
            framework_adapter, class_name, None
        )
        self.params: Dict[str, ndarray] = framework_adapter_plugin.get_tensor_dict(
            model_interface.provide_model()
        )
