# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cutoff time based Straggler Handling function."""
import time

import numpy as np

from openfl.component.straggler_handling_functions.straggler_handling_function import (
    StragglerHandlingFunction,
)


class CutoffTimeBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(
        self, round_start_time=None, straggler_cutoff_time=np.inf, minimum_reporting=1, **kwargs
    ):
        self.round_start_time = round_start_time
        self.straggler_cutoff_time = straggler_cutoff_time
        self.minimum_reporting = minimum_reporting

    def straggler_time_expired(self):
        return self.round_start_time is not None and (
            (time.time() - self.round_start_time) > self.straggler_cutoff_time
        )

    def minimum_collaborators_reported(self, num_collaborators_done):
        return num_collaborators_done >= self.minimum_reporting

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators=None):
        cutoff = self.straggler_time_expired() and self.minimum_collaborators_reported(
            num_collaborators_done
        )
        return cutoff
