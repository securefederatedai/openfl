# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cutoff time based Straggler Handling function."""
import numpy as np
import time

from openfl.component.straggler_handling_functions import StragglerHandlingFunction


class CutoffTimeBasedStragglerHandling(StragglerHandlingFunction):
    """Cutoff time based Straggler Handling function."""
    
    def __init__(
        self,
        round_start_time=None,
        straggler_cutoff_time=np.inf,
        minimum_reporting=1,
        **kwargs
    ):
        """Initialize a CutoffTimeBasedStragglerHandling object.

        Args:
            round_start_time (optional): The start time of the round. Defaults to None.
            straggler_cutoff_time (float, optional): The cutoff time for stragglers. Defaults to np.inf.
            minimum_reporting (int, optional): The minimum number of collaborators that should report. Defaults to 1.
            **kwargs: Variable length argument list.
        """
        self.round_start_time = round_start_time
        self.straggler_cutoff_time = straggler_cutoff_time
        self.minimum_reporting = minimum_reporting

    def straggler_time_expired(self):
        """Check if the straggler time has expired.

        Returns:
            bool: True if the straggler time has expired, False otherwise.
        """
        return self.round_start_time is not None and (
            (time.time() - self.round_start_time) > self.straggler_cutoff_time)

    def minimum_collaborators_reported(self, num_collaborators_done):
        """Check if the minimum number of collaborators have reported.

        Args:
            num_collaborators_done (int): The number of collaborators that have reported.

        Returns:
            bool: True if the minimum number of collaborators have reported, False otherwise.
        """
        return num_collaborators_done >= self.minimum_reporting

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators=None):
        """Check if the straggler cutoff conditions are met.

        Args:
            num_collaborators_done (int): The number of collaborators that have reported.
            all_collaborators (optional): All the collaborators. Defaults to None.

        Returns:
            bool: True if the straggler cutoff conditions are met, False otherwise.
        """
        cutoff = self.straggler_time_expired() and self.minimum_collaborators_reported(
            num_collaborators_done)
        return cutoff
