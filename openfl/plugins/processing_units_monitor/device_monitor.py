# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Device monitor plugin module."""


class DeviceMonitor:
    """Device monitor plugin interface."""

    def get_driver_version(self) -> str:
        """Get device's driver version."""
        raise NotImplementedError

    def get_device_utilization(self, index: int) -> str:
        """
        Get device utilization method.

        It is just a general method that returns a string that may be shown to the frontend user.
        """
        raise NotImplementedError
