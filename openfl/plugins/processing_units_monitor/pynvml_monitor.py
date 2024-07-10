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

"""
pynvml CUDA Device monitor plugin module.

Required package: pynvml
"""

import pynvml

from openfl.plugins.processing_units_monitor.cuda_device_monitor import CUDADeviceMonitor

pynvml.nvmlInit()


class PynvmlCUDADeviceMonitor(CUDADeviceMonitor):
    """CUDA Device monitor plugin using pynvml lib."""

    def __init__(self) -> None:
        """Initialize pynvml plugin."""
        super().__init__()

    def get_driver_version(self) -> str:
        """Get Nvidia driver version."""
        return pynvml.nvmlSystemGetDriverVersion().decode("utf-8")

    def get_device_memory_total(self, index: int) -> int:
        """Get total memory available on the device."""
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total

    def get_device_memory_utilized(self, index: int) -> int:
        """Get utilized memory on the device."""
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used

    def get_device_utilization(self, index: int) -> str:
        """
        Get device utilization.

        It is just a general method that returns a string that may be shown to the frontend user.
        """
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return f"{info_utilization.gpu}%"

    def get_device_name(self, index: int) -> str:
        """Get device utilization method."""
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        device_name = pynvml.nvmlDeviceGetName(handle)
        return device_name

    def get_cuda_version(self) -> str:
        """
        Get CUDA driver version.

        The CUDA version is specified as (1000 * major + 10 * minor),
        so CUDA 11.2 should be specified as 11020.
        https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html
        """
        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        major_version = int(cuda_version / 1000)
        minor_version = int(cuda_version % 1000 / 10)
        return f"{major_version}.{minor_version}"
