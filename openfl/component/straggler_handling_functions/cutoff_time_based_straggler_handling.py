# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Cutoff time based Straggler Handling function."""
import threading
import time
from logging import getLogger
from typing import Callable

import numpy as np

from openfl.component.straggler_handling_functions.straggler_handling_function import (
    StragglerHandlingPolicy,
)


class CutoffTimeBasedStragglerHandling(StragglerHandlingPolicy):
    def __init__(
        self, round_start_time=None, straggler_cutoff_time=np.inf, minimum_reporting=1, **kwargs
    ):
        """
         Initialize a CutoffTimeBasedStragglerHandling object.

        Args:
            round_start_time (optional): The start time of the round. Defaults
                to None.
            straggler_cutoff_time (float, optional): The cutoff time for
                stragglers. Defaults to np.inf.
            minimum_reporting (int, optional): The minimum number of
                collaborators that should report. Defaults to 1.
            **kwargs: Variable length argument list.
        """
        if minimum_reporting <= 0:
            raise ValueError("minimum_reporting must be >0")

        self.round_start_time = round_start_time
        self.straggler_cutoff_time = straggler_cutoff_time
        self.minimum_reporting = minimum_reporting
        self.logger = getLogger(__name__)

        if self.straggler_cutoff_time == np.inf:
            self.logger.warning(
                "CutoffTimeBasedStragglerHandling is disabled as straggler_cutoff_time "
                "is set to np.inf."
            )

    def reset_policy_for_round(self) -> None:
        """
        Reset timer for the next round.
        """
        if hasattr(self, "timer"):
            self.timer.cancel()
            delattr(self, "timer")

    def start_policy(self, callback: Callable) -> None:
        """
        Start time-based straggler handling policy for collaborator for
        a particular round.

        Args:
            callback: Callable
                Callback function for when straggler_cutoff_time elapses

        Returns:
            None
        """
        # If straggler_cutoff_time is set to infinity
        # or if the timer is already running,
        # do not start the policy.
        if self.straggler_cutoff_time == np.inf or hasattr(self, "timer"):
            return
        self.round_start_time = time.time()
        self.timer = threading.Timer(
            self.straggler_cutoff_time,
            callback,
        )
        self.timer.daemon = True
        self.timer.start()

    def straggler_cutoff_check(
        self,
        num_collaborators_done: int,
        num_all_collaborators: int,
    ) -> bool:
        """
        If minimum_reporting collaborators have reported results within
        straggler_cutoff_time then return True, otherwise False.

        Args:
            num_collaborators_done: int
                Number of collaborators finished.
            num_all_collaborators: int
                Total number of collaborators.

        Returns:
            bool
        """

        # if straggler time has not expired then
        # wait for ALL collaborators to report results.
        if not self.__straggler_time_expired():
            return num_all_collaborators == num_collaborators_done

        # Check if time has expired
        elif self.__straggler_time_expired():
            # Check if minimum_reporting collaborators have reported results
            if self.__minimum_collaborators_reported(num_collaborators_done):
                self.logger.info(
                    f"{num_collaborators_done} collaborators reported results within "
                    "cutoff time. Applying cutoff policy and proceeding with end of round."
                )
                return True
            self.logger.info(
                "Disregarded straggler handling policy and waiting for minimum "
                f"{self.minimum_reporting} collaborator(s) to report results."
            )
            return False

        # Something has gone wrong, unhandled scenario, raising error.
        raise ValueError(
            "Unhandled scenario"
            f"{self.__straggler_time_expired()=}"
            f"{self.straggler_cutoff_time=}"
        )

    def __straggler_time_expired(self) -> bool:
        """Check if the straggler time has expired.

        Returns:
            bool: True if the straggler time has expired, False otherwise.
        """
        return self.round_start_time is not None and (
            (time.time() - self.round_start_time) > self.straggler_cutoff_time
        )

    def __minimum_collaborators_reported(self, num_collaborators_done) -> bool:
        """Check if the minimum number of collaborators have reported.

        Args:
            num_collaborators_done (int): The number of collaborators that
                have reported.

        Returns:
            bool: True if the minimum number of collaborators have reported,
                False otherwise.
        """
        return num_collaborators_done >= self.minimum_reporting
