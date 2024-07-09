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

"""Serializer plugin interface."""


class Serializer:
    """Serializer API plugin."""

    def __init__(self) -> None:
        """Initialize serializer."""
        pass

    @staticmethod
    def serialize(object_, filename):
        """Serialize an object and save to disk."""
        raise NotImplementedError

    @staticmethod
    def restore_object(filename):
        """Load and deserialize an object."""
        raise NotImplementedError
