# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Percentage based Straggler Handling function."""
from openfl.component.straggler_handling_functions import (
    StragglerHandlingFunction,
)


class PercentageBasedStragglerHandling(StragglerHandlingFunction):
    """Percentage based Straggler Handling function."""

    def __init__(self,
                 percent_collaborators_needed=1.0,
                 minimum_reporting=1,
                 **kwargs):
        """Initialize a PercentageBasedStragglerHandling object.

        Args:
            percent_collaborators_needed (float, optional): The percentage of
                collaborators needed. Defaults to 1.0.
            minimum_reporting (int, optional): The minimum number of
                collaborators that should report. Defaults to 1.
            **kwargs: Variable length argument list.
        """
        self.percent_collaborators_needed = percent_collaborators_needed
        self.minimum_reporting = minimum_reporting

    def minimum_collaborators_reported(self, num_collaborators_done):
        """Check if the minimum number of collaborators have reported.

        Args:
            num_collaborators_done (int): The number of collaborators that
                have reported.

        Returns:
            bool: True if the minimum number of collaborators have reported,
                False otherwise.
        """
        return num_collaborators_done >= self.minimum_reporting

    def straggler_cutoff_check(self, num_collaborators_done,
                               all_collaborators):
        """Check if the straggler cutoff conditions are met.

        Args:
            num_collaborators_done (int): The number of collaborators that
                have reported.
            all_collaborators (list): All the collaborators.

        Returns:
            bool: True if the straggler cutoff conditions are met, False
                otherwise.
        """
        cutoff = (
            num_collaborators_done
            >= self.percent_collaborators_needed * len(all_collaborators)
        ) and self.minimum_collaborators_reported(num_collaborators_done)
        return cutoff
