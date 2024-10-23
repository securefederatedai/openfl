# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Straggler handling module."""

import threading
import time
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Callable

import numpy as np


class StragglerHandlingPolicy(ABC):
    """Federated Learning straggler handling interface."""

    @abstractmethod
    def start_policy(self, **kwargs) -> None:
        """
        Start straggler handling policy for collaborator for a particular round.
        NOTE: Refer CutoffPolicy for reference.

        Args:
            **kwargs

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def reset_policy_for_round(self) -> None:
        """
        Reset policy variable for the next round.

        Args:
            None

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def straggler_cutoff_check(
        self, num_collaborators_done: int, num_all_collaborators: int, **kwargs
    ) -> bool:
        """
        Determines whether it is time to end the round early.

        Args:
            num_collaborators_done: int
                Number of collaborators finished.
            num_all_collaborators: int
                Total number of collaborators.

        Returns:
            bool: True if it is time to end the round early, False otherwise.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError


class CutoffPolicy(StragglerHandlingPolicy):
    """Cutoff time based Straggler Handling function."""

    def __init__(
        self, round_start_time=None, straggler_cutoff_time=np.inf, minimum_reporting=1, **kwargs
    ):
        """
         Initialize a CutoffPolicy object.

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
        self.is_timer_started = False
        self.logger = getLogger(__name__)

        if self.straggler_cutoff_time == np.inf:
            self.logger.warning(
                "CutoffPolicy is disabled as straggler_cutoff_time " "is set to np.inf."
            )

    def reset_policy_for_round(self) -> None:
        """
        Reset timer for the next round.
        """
        if hasattr(self, "timer"):
            self.timer.cancel()
        self.is_timer_started = False

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
        if self.straggler_cutoff_time == np.inf or self.is_timer_started:
            return

        self.round_start_time = time.time()
        self.timer = threading.Timer(
            self.straggler_cutoff_time,
            callback,
        )
        self.timer.daemon = True
        self.timer.start()
        self.is_timer_started = True

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
            bool: True if the straggler cutoff conditions are met, False otherwise.
        """

        # if straggler time has not expired then
        # wait for ALL collaborators to report results.
        if not self.__straggler_time_expired():
            return num_all_collaborators == num_collaborators_done

        # Time has expired
        # Check if minimum_reporting collaborators have reported results
        elif self.__minimum_collaborators_reported(num_collaborators_done):
            self.logger.info(
                f"{num_collaborators_done} collaborators have reported results. "
                "Applying cutoff policy and proceeding with end of round."
            )
            return True
        else:
            self.logger.info(
                f"Waiting for minimum {self.minimum_reporting} collaborator(s) to report results."
            )
            return False

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


class PercentagePolicy(StragglerHandlingPolicy):
    """Percentage based Straggler Handling function."""

    def __init__(self, percent_collaborators_needed=1.0, minimum_reporting=1, **kwargs):
        """Initialize a PercentagePolicy object.

        Args:
            percent_collaborators_needed (float, optional): The percentage of
                collaborators needed. Defaults to 1.0.
            minimum_reporting (int, optional): The minimum number of
                collaborators that should report. Defaults to 1.
            **kwargs: Variable length argument list.
        """
        if minimum_reporting <= 0:
            raise ValueError("minimum_reporting must be >0")

        self.percent_collaborators_needed = percent_collaborators_needed
        self.minimum_reporting = minimum_reporting
        self.logger = getLogger(__name__)

    def reset_policy_for_round(self) -> None:
        """
        Not required in PercentagePolicy.
        """
        pass

    def start_policy(self, **kwargs) -> None:
        """
        Not required in PercentagePolicy.
        """
        pass

    def straggler_cutoff_check(
        self,
        num_collaborators_done: int,
        num_all_collaborators: int,
    ) -> bool:
        """
        If percent_collaborators_needed and minimum_reporting collaborators have
        reported results, then it is time to end round early.

        Args:
            num_collaborators_done (int): The number of collaborators that
                have reported.
            all_collaborators (list): All the collaborators.

        Returns:
            bool: True if the straggler cutoff conditions are met, False
                otherwise.
        """
        return (
            num_collaborators_done >= self.percent_collaborators_needed * num_all_collaborators
        ) and self.__minimum_collaborators_reported(num_collaborators_done)

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
