# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cutoff time based Straggler Handling function."""
import numpy as np
import time
import threading
from typing import Callable
from logging import getLogger

from openfl.component.straggler_handling_functions import StragglerHandlingFunction


class CutoffTimeBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(
        self,
        round_start_time=None,
        straggler_cutoff_time=np.inf,
        minimum_reporting=1,
        **kwargs
    ):
        if minimum_reporting == 0: raise ValueError("minimum_reporting cannot be 0")

        self.round_start_time = round_start_time
        self.straggler_cutoff_time = straggler_cutoff_time
        self.minimum_reporting = minimum_reporting
        self.__is_policy_applied = False
        self.logger = getLogger(__name__)

        if self.straggler_cutoff_time == np.inf:
            self.logger.warning(
                "CutoffTimeBasedStragglerHandling is disabled as straggler_cutoff_time "
                "is set to np.inf."
            )

    def reset_policy_for_round(self) -> None:
        """
        Control whether to start the timer or not.
        """
        self.__is_policy_applied = False

    def start_policy(
        self, callback: Callable, collaborator_name: str
    ) -> None:
        """
        Start time-based straggler handling policy for collaborator for
        a particular round.

        Args:
            callback: Callable
                Callback function for when straggler_cutoff_time elapses
            collaborator_name: str
                Name of the collaborator

        Returns:
            None
        """
        # If straggler_cutoff_time is set to infinite or
        # if the timer already expired for the current round do not start
        # the timer again until next round.
        if self.straggler_cutoff_time == np.inf or self.__is_policy_applied:
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
        self, num_collaborators_done, all_collaborators=None,
    ) -> bool:
        """
        If minimum_reporting collaborators have reported results within
        straggler_cutoff_time, then return True otherwise False.
        """
        if not self.__straggler_time_expired():
            return False
        elif self.__straggler_time_expired() and not self.__is_policy_applied:
            self.__is_policy_applied = True
            if self.__minimum_collaborators_reported(num_collaborators_done):
                self.logger.info(
                    f"{len(self.collaborators_done)} collaborators reported results within cutoff "
                    f"time. Applying cutoff policy and proceeding with end of round."
                )
                return True
            else:
                self.logger.info(
                    "Disregarding straggler handling policy and waiting for ALL "
                    f"{len(all_collaborators)} collaborator(s) to report results."
                )
                return False
        elif self.__straggler_time_expired() and self.__is_policy_applied:
            return len(all_collaborators) == num_collaborators_done
        else:
            self.logger.info("*"*20)
            self.logger.info("Something has gone horribly wrong, and needs to be looked at immediately...")
            self.logger.info("*"*20)
            return None
        # cutoff = self.__straggler_time_expired() and self.__minimum_collaborators_reported(
        #     num_collaborators_done)
        # return cutoff

    def __straggler_time_expired(self):
        """
        Determines if straggler_cutoff_time is elapsed
        """
        return self.round_start_time is not None and (
            (time.time() - self.round_start_time) > self.straggler_cutoff_time)

    def __minimum_collaborators_reported(self, num_collaborators_done):
        """
        If minimum required collaborators have reported results, then return True
        otherwise False.
        """
        return num_collaborators_done >= self.minimum_reporting
