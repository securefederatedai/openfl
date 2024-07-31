# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl common object types."""

from abc import ABCMeta
from collections import namedtuple

TensorKey = namedtuple("TensorKey", ["tensor_name", "origin", "round_number", "report", "tags"])
TaskResultKey = namedtuple("TaskResultKey", ["task_name", "owner", "round_number"])

Metric = namedtuple("Metric", ["name", "value"])
LocalTensor = namedtuple("LocalTensor", ["col_name", "tensor", "weight"])


class SingletonABCMeta(ABCMeta):
    """Metaclass for singleton instances.

    This metaclass ensures that only one instance of any class using it can be
    created.

    Attributes:
        _instances (dict): A dictionary mapping classes to their instances.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Use the singleton instance if it has already been created.

        Args:
            *args: Positional arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            Any: The singleton instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
