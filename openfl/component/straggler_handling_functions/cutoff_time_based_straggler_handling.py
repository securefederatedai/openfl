# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cutoff time based Straggler Handling function."""
import numpy as np
import time
import threading
from typing import Callable
from logging import getLogger

from openfl.component.straggler_handling_functions import StragglerHandlingPolicy


class CutoffTimeBasedStragglerHandling(StragglerHandlingPolicy):
    def __init__(
        self,
        round_start_time=None,
        straggler_cutoff_time=np.inf,
        minimum_reporting=1,
        **kwargs
    ):
        if minimum_reporting <= 0:
            raise ValueError(f"minimum_reporting cannot be {minimum_reporting}")

        self.round_start_time = round_start_time
        self.straggler_cutoff_time = straggler_cutoff_time
        self.minimum_reporting = minimum_reporting
        self.__is_policy_applied_for_round = False
        self.logger = getLogger(__name__)

        if self.straggler_cutoff_time == np.inf:
            self.logger.warning(
                "CutoffTimeBasedStragglerHandling is disabled as straggler_cutoff_time "
                "is set to np.inf."
            )

    def reset_policy_for_round(self) -> None:
        """
        Reset policy variable for the next round.
        """
        self.__is_policy_applied_for_round = False

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
        # If straggler_cutoff_time is set to infinite or
        # if the timer already expired for the current round do not start
        # the timer again until next round.
        if self.straggler_cutoff_time == np.inf or self.__is_policy_applied_for_round:
            return
        self.round_start_time = time.time()
        if hasattr(self, "timer"):
            self.timer.cancel()
            delattr(self, "timer")
        self.timer = threading.Timer(
            self.straggler_cutoff_time, callback,
        )
        self.timer.daemon = True
        self.timer.start()

    def straggler_cutoff_check(
        self, num_collaborators_done: int, num_all_collaborators: int,
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
        if not self.__straggler_time_expired():
            return False
        # Check if time has expired and policy is not applied
        elif self.__straggler_time_expired() and not self.__is_policy_applied_for_round:
            # Stop timer from restarting for current round and
            # if set to True wait for ALL collaborators instead of minimum_reporting
            self.__is_policy_applied_for_round = True
            # Check if minimum_reporting collaborators have reported results
            if self.__minimum_collaborators_reported(num_collaborators_done):
                self.logger.info(
                    f"{num_collaborators_done} collaborators reported results within "
                    "cutoff time. Applying cutoff policy and proceeding with end of round."
                )
                return True
            self.logger.info(
                "Disregarding straggler handling policy and waiting for ALL "
                f"{num_all_collaborators} collaborator(s) to report results."
            )
            return False
        # If straggler_cutoff_time is set to infinite,
        # OR
        # If minimum_reporting collaborators have not reported results within cutoff
        # time,
        # then disregard the policy and wait for ALL collaborators to report
        # results.
        elif (
            self.straggler_cutoff_time == np.inf
            or (self.__straggler_time_expired() and self.__is_policy_applied_for_round)
        ):
            return num_all_collaborators == num_collaborators_done
        # Something has gone, unhandled scenario, raising error.
        raise ValueError("Unhandled scenario")

    def __straggler_time_expired(self) -> bool:
        """
        Determines if straggler_cutoff_time is elapsed.
        """
        return (
            self.round_start_time is not None
            and ((time.time() - self.round_start_time) > self.straggler_cutoff_time)
        )

    def __minimum_collaborators_reported(self, num_collaborators_done) -> bool:
        """
        If minimum required collaborators have reported results, then return True
        otherwise False.
        """
        return num_collaborators_done >= self.minimum_reporting
