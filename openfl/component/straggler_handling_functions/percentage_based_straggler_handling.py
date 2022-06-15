# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Percentage based Straggler Handling function."""
from logging import getLogger

from openfl.component.straggler_handling_functions import StragglerHandlingFunction

class PercentageBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(self, percent_collaborators_needed):
        self.percent_collaborators_needed = percent_collaborators_needed
        self.logger = getLogger(__name__)

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators):
        cutoff = (num_collaborators_done >= self.percent_collaborators_needed*all_collaborators)
        return cutoff
