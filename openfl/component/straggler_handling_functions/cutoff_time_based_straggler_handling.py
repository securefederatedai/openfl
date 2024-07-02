# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cutoff time based Straggler Handling function."""
import numpy as np
import time
import threading
from typing import Callable
from logging import Logger

from openfl.component.straggler_handling_functions import StragglerHandlingFunction


class CutoffTimeBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(
        self,
        round_start_time=None,
        straggler_cutoff_time=np.inf,
        minimum_reporting=1,
        **kwargs
    ):
        self.round_start_time = round_start_time
        self.straggler_cutoff_time = straggler_cutoff_time
        self.minimum_reporting = minimum_reporting

    def start_policy(
        self, callback: Callable, logger: Logger, collaborator_name: str
    ) -> None:
        """
        Start policy.

        Args:
            callback: Callable
                Callback function for when straggler_cutoff_time elapses
            logger: Logger

        Returns:
            None
        """
        self.round_start_time = time.time()
        if self.straggler_cutoff_time == np.inf:
            logger.warning("straggler_cutoff_time is set to np.inf, timer will not start.")
            return
        if hasattr(self, "timer"):
            self.timer.cancel()
            delattr(self, "timer")
        self.timer = threading.Timer(
            self.straggler_cutoff_time, callback,
        )
        self.timer.daemon = True
        self.timer.start()

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators=None):
        cutoff = self.__straggler_time_expired() and self.__minimum_collaborators_reported(
            num_collaborators_done)
        return cutoff

    def __straggler_time_expired(self):
        return self.round_start_time is not None and (
            (time.time() - self.round_start_time) > self.straggler_cutoff_time)

    def __minimum_collaborators_reported(self, num_collaborators_done):
        return num_collaborators_done >= self.minimum_reporting
