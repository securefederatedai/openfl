# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom model numpy adapter."""
from typing import Any
from typing import Dict

from openfl.plugins.frameworks_adapters.framework_adapter_interface import (
    FrameworkAdapterPluginInterface,
)


class CustomFrameworkAdapter(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    @staticmethod
    def get_tensor_dict(model: Any, optimizer: Any = None) -> Dict:
        """Extract tensors from a model."""
        return {'w': model.weights}

    @staticmethod
    def set_tensor_dict(model: Any, tensor_dict: Any,
                        optimizer: Any = None, device: Any = 'cpu') -> Any:
        """Load tensors to a model."""
        model.weights = tensor_dict['w']
