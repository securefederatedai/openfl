# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Straggler handling module."""

from abc import ABC
from abc import abstractmethod
from typing import Callable
from logging import Logger


class StragglerHandlingFunction(ABC):
    """Federated Learning straggler handling interface."""

    @abstractmethod
    def start_policy(
        self, callback: Callable, collaborator_name: str
    ) -> None:
        """
        Start straggler handling policy for collaborator for a particular round.
        NOTE: Refer CutoffTimeBasedStragglerHandling for reference.

        Args:
            callback: Callable
                Callback function for when straggler_cutoff_time elapses
            collaborator_name: str
                Name of the collaborator

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def straggler_cutoff_check(self, **kwargs):
        """
        Determines whether it is time to end the round early.
        Returns:
            bool
        """
        raise NotImplementedError
