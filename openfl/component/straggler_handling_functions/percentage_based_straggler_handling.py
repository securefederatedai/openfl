# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Percentage based Straggler Handling function."""
from logging import getLogger
from openfl.component.straggler_handling_functions import StragglerHandlingPolicy


class PercentageBasedStragglerHandling(StragglerHandlingPolicy):
    def __init__(
        self,
        percent_collaborators_needed=1.0,
        minimum_reporting=1,
        **kwargs
    ):
        if minimum_reporting <= 0:
            raise ValueError(f"minimum_reporting cannot be {minimum_reporting}")

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
        self, num_collaborators_done: int, num_all_collaborators: int,
    ) -> bool:
        """
        If percent_collaborators_needed and minimum_reporting collaborators have
        reported results, then it is time to end round early.
        """
        return (
            (num_collaborators_done >= self.percent_collaborators_needed * num_all_collaborators)
            and self.__minimum_collaborators_reported(num_collaborators_done)
        )

    def __minimum_collaborators_reported(self, num_collaborators_done) -> bool:
        """
        If minimum required collaborators have reported results, then return True
        otherwise False.
        """
        return num_collaborators_done >= self.minimum_reporting
