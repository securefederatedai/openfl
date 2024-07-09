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

"""Framework Adapter plugin interface."""


class FrameworkAdapterPluginInterface:
    """Framework adapter plugin class."""

    def __init__(self) -> None:
        """Initialize framework adapter."""
        pass

    @staticmethod
    def serialization_setup():
        """Prepare model for serialization (optional)."""
        pass

    @staticmethod
    def get_tensor_dict(model, optimizer=None) -> dict:
        """
        Extract tensor dict from a model and an optimizer.

        Returns:
        dict {weight name: numpy ndarray}
        """
        raise NotImplementedError

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device="cpu"):
        """
        Set tensor dict from a model and an optimizer.

        Given a dict {weight name: numpy ndarray} sets weights to
        the model and optimizer objects inplace.
        """
        raise NotImplementedError
