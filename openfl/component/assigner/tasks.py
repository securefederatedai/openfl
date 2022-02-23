# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Task module."""


from dataclasses import dataclass
from dataclasses import field


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
