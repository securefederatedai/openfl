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

"""CUDA Device monitor plugin module."""

from openfl.plugins.processing_units_monitor.device_monitor import DeviceMonitor


class CUDADeviceMonitor(DeviceMonitor):
    """CUDA Device monitor plugin."""

    def get_driver_version(self) -> str:
        """Get CUDA driver version."""
        raise NotImplementedError

    def get_device_memory_total(self, index: int) -> int:
        """Get total memory available on the device."""
        raise NotImplementedError

    def get_device_memory_utilized(self, index: int) -> int:
        """Get utilized memory on the device."""
        raise NotImplementedError

    def get_device_utilization(self, index: int) -> str:
        """
        Get device utilization.

        It is just a general method that returns a string that may be shown to the frontend user.
        """
        raise NotImplementedError

    def get_device_name(self, index: int) -> str:
        """Get device name."""
        raise NotImplementedError

    def get_cuda_version(self) -> str:
        """Get CUDA driver version."""
        raise NotImplementedError
