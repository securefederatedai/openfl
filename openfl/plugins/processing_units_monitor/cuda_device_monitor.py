# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CUDA Device monitor plugin module."""

from .device_monitor import DeviceMonitor


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
