# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cutoff time based Straggler Handling function."""
from logging import getLogger
import time

from openfl.component.straggler_handling_functions import StragglerHandlingFunction

class CutoffTimeBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(self, round_start_time, straggler_cutoff_time, minimum_reporting):
        self.round_start_time = round_start_time
        self.straggler_cutoff_time = straggler_cutoff_time
        self.minimum_reporting = minimum_reporting
        self.logger = getLogger(__name__)

    def straggler_time_expired(self):
        return self.round_start_time is not None and ((time.time() - self.round_start_time) > self.straggler_cutoff_time)

    def minimum_collaborators_reported(self, num_collaborators_done):
        return num_collaborators_done >= self.minimum_reporting

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators=None):
        cutoff = self.straggler_time_expired() and self.minimum_collaborators_reported(num_collaborators_done)
        return cutoff
