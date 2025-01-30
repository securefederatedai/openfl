# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""CUDA Device monitor plugin module."""

from openfl.plugins.processing_units_monitor.device_monitor import DeviceMonitor


class CUDADeviceMonitor(DeviceMonitor):
    """CUDA Device monitor plugin."""

    def get_driver_version(self) -> str:
        """Get CUDA driver version.

        This method is not implemented.

        Returns:
            str: The CUDA driver version.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    def get_device_memory_total(self, index: int) -> int:
        """Get total memory available on the device.

        This method is not implemented.

        Args:
            index (int): The index of the device.

        Returns:
            int: The total memory available on the device.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    def get_device_memory_utilized(self, index: int) -> int:
        """Get utilized memory on the device.

        This method is not implemented.

        Args:
            index (int): The index of the device.

        Returns:
            int: The utilized memory on the device.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    def get_device_utilization(self, index: int) -> str:
        """Get device utilization.

        It is just a general method that returns a string that may be shown to
        the frontend user.
        This method is not implemented.

        Args:
            index (int): The index of the device.

        Returns:
            str: The device utilization.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    def get_device_name(self, index: int) -> str:
        """Get device name.

        This method is not implemented.

        Args:
            index (int): The index of the device.

        Returns:
            str: The device name.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    def get_cuda_version(self) -> str:
        """Get CUDA driver version.

        This method is not implemented.

        Returns:
            str: The CUDA driver version.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError
