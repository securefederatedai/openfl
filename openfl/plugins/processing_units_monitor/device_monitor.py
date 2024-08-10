# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Device monitor plugin module."""


class DeviceMonitor:
    """Device monitor plugin interface."""

    def get_driver_version(self) -> str:
        """Get device's driver version.

        This method is not implemented.

        Returns:
            str: The device's driver version.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError

    def get_device_utilization(self, index: int) -> str:
        """Get device utilization method.

        It is just a general method that returns a string that may be shown to
        the frontend user.

        Args:
            index (int): The index of the device.

        Returns:
            str: The device utilization.

        Raises:
            NotImplementedError: This is a placeholder method that needs to be
                implemented in subclasses.
        """
        raise NotImplementedError
