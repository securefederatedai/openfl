# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        class_name = splitext(model_interface.framework_plugin)[1].strip(".")
        module_path = splitext(model_interface.framework_plugin)[0]
        framework_adapter = import_module(module_path)
        framework_adapter_plugin: FrameworkAdapterPluginInterface = getattr(
            framework_adapter, class_name, None
        )
        self.params: Dict[str, ndarray] = framework_adapter_plugin.get_tensor_dict(
            model_interface.provide_model()
        )
