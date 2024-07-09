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
