# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
pynvml CUDA Device monitor plugin module.

Required package: pynvml
"""

import pynvml

from .cuda_device_monitor import CUDADeviceMonitor

pynvml.nvmlInit()


class PynvmlCUDADeviceMonitor(CUDADeviceMonitor):
    """CUDA Device monitor plugin using pynvml lib."""

    def __init__(self) -> None:
        """Initialize pynvml plugin."""
        super().__init__()

    def get_driver_version(self) -> str:
        """Get Nvidia driver version."""
        return pynvml.nvmlSystemGetDriverVersion().decode('utf-8')

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
        return f'{info_utilization.gpu}%'

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
        return f'{major_version}.{minor_version}'
