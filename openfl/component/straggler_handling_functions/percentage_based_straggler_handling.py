# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Percentage based Straggler Handling function."""
from logging import getLogger
from typing import Callable
from openfl.component.straggler_handling_functions import StragglerHandlingFunction


class PercentageBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(
        self,
        percent_collaborators_needed=1.0,
        minimum_reporting=1,
        **kwargs
    ):
        if minimum_reporting == 0: raise ValueError("minimum_reporting cannot be 0")

        self.percent_collaborators_needed = percent_collaborators_needed
        self.minimum_reporting = minimum_reporting
        self.logger = getLogger(__name__)

    def start_policy(
        self, callback: Callable, collaborator_name: str
    ) -> None:
        """
        Not required in PercentageBasedStragglerHandling.
        """
        pass

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators):
        """
        If percent_collaborators_needed and minimum_reporting collaborators have
        reported results, then it is time to end round early.
        """
        cutoff = (num_collaborators_done >= self.percent_collaborators_needed * len(
            all_collaborators)) and self.__minimum_collaborators_reported(num_collaborators_done)
        return cutoff

    def __minimum_collaborators_reported(self, num_collaborators_done):
        """
        If minimum required collaborators have reported results, then return True
        otherwise False.
        """
        return num_collaborators_done >= self.minimum_reporting
