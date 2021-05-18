# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl common object types."""

from collections import namedtuple

TensorKey = namedtuple('TensorKey', ['tensor_name', 'origin', 'round_number', 'report', 'tags'])
TaskResultKey = namedtuple('TaskResultKey', ['task_name', 'owner', 'round_number'])

Metric = namedtuple('Metric', ['name', 'value'])
LocalTensor = namedtuple('LocalTensor', ['col_name', 'tensor', 'weight'])


class Singleton:
    """Metaclass for singleton instances."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Use the singleton instance if it has already been created."""
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance
