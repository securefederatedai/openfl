# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Percentage based Straggler Handling function."""
from openfl.component.straggler_handling_functions import StragglerHandlingFunction


class PercentageBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(
        self,
        percent_collaborators_needed=1.0,
        minimum_reporting=1,
        **kwargs):
        self.percent_collaborators_needed = percent_collaborators_needed
        self.minimum_reporting = minimum_reporting

    def minimum_collaborators_reported(self, num_collaborators_done):
        return num_collaborators_done >= self.minimum_reporting

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators):
        cutoff = (num_collaborators_done >= self.percent_collaborators_needed * len(
            all_collaborators)) and self.minimum_collaborators_reported(num_collaborators_done)
        return cutoff
