# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Straggler handling module."""

from abc import ABC
from abc import abstractmethod


class StragglerHandlingFunction(ABC):
    """Federated Learning straggler handling interface."""

    @abstractmethod
    def straggler_cutoff_check(self, **kwargs):
        """
        Determines whether it is time to end the round early.
        Returns:
            bool
        """
        raise NotImplementedError
