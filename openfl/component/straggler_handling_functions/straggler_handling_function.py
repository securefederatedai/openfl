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
    def straggler_cutoff_check(self, **kwargs):
        """
        Determines whether it is time to end the round early.
        Returns:
            bool
        """
        raise NotImplementedError

    @abstractmethod
    def start_straggler_cutoff_timer(
        self, callback: Callable, logger: Logger
    ) -> None:
        """
        Start timer.

        Args:
            callback: Callable
                Callback function for when straggler_cutoff_time elapses
            logger: Logger

        Returns:
            None
        """
        raise NotImplementedError
