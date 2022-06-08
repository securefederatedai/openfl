# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Straggler handling module."""

from abc import ABC
from abc import abstractmethod

class StragglerHandlingFunction(ABC):

    @abstractmethod
    def straggler_cutoff_check(self, **kwargs):
        raise NotImplementedError
