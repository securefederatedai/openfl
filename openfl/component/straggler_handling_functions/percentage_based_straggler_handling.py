# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Percentage based Straggler Handling function."""
from logging import getLogger

from openfl.component.straggler_handling_functions.straggler_handling_function import (
    StragglerHandlingPolicy,
)


class PercentageBasedStragglerHandling(StragglerHandlingPolicy):
    """Percentage based Straggler Handling function."""

    def __init__(self, percent_collaborators_needed=1.0, minimum_reporting=1, **kwargs):
        """Initialize a PercentageBasedStragglerHandling object.

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
        Not required in PercentageBasedStragglerHandling.
        """
        pass

    def start_policy(self, **kwargs) -> None:
        """
        Not required in PercentageBasedStragglerHandling.
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
