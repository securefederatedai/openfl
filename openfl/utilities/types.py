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

"""openfl common object types."""

from abc import ABCMeta
from collections import namedtuple

TensorKey = namedtuple(
    "TensorKey", ["tensor_name", "origin", "round_number", "report", "tags"]
)
TaskResultKey = namedtuple(
    "TaskResultKey", ["task_name", "owner", "round_number"]
)

Metric = namedtuple("Metric", ["name", "value"])
LocalTensor = namedtuple("LocalTensor", ["col_name", "tensor", "weight"])


class SingletonABCMeta(ABCMeta):
    """Metaclass for singleton instances."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Use the singleton instance if it has already been created."""
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]
