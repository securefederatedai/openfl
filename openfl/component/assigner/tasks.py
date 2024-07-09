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

"""Task module."""


from dataclasses import dataclass, field


@dataclass
class Task:
    """Task base dataclass."""

    name: str
    function_name: str
    task_type: str
    apply_local: bool = False
    parameters: dict = field(default_factory=dict)  # We can expend it in the future


@dataclass
class TrainTask(Task):
    """TrainTask class."""

    task_type: str = 'train'


@dataclass
class ValidateTask(Task):
    """Validation Task class."""

    task_type: str = 'validate'
