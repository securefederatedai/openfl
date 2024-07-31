# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Task module."""

from dataclasses import dataclass, field


@dataclass
class Task:
    """Task base dataclass.

    Args:
        name (str): Name of the task.
        function_name (str): Name of the function to be executed for the task.
        task_type (str): Type of the task.
        apply_local (bool, optional): Whether to apply the task locally.
            Defaults to False.
        parameters (dict, optional): Parameters for the task. Defaults to an
            empty dictionary.
    """

    name: str
    function_name: str
    task_type: str
    apply_local: bool = False
    parameters: dict = field(default_factory=dict)  # We can expend it in the future


@dataclass
class TrainTask(Task):
    """TrainTask class.

    Args:
        name (str): Name of the task.
        function_name (str): Name of the function to be executed for the task.
        apply_local (bool, optional): Whether to apply the task locally.
            Defaults to False.
        parameters (dict, optional): Parameters for the task. Defaults to an
            empty dictionary.

    Attributes:
        task_type (str): Type of the task. Set to 'train'.
    """

    task_type: str = "train"


@dataclass
class ValidateTask(Task):
    """Validation Task class.

    Args:
        name (str): Name of the task.
        function_name (str): Name of the function to be executed for the task.
        apply_local (bool, optional): Whether to apply the task locally.
            Defaults to False.
        parameters (dict, optional): Parameters for the task. Defaults to an
            empty dictionary.

    Attributes:
        task_type (str): Type of the task. Set to 'validate'.
    """

    task_type: str = "validate"
