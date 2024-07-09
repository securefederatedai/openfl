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

"""Percentage based Straggler Handling function."""
from openfl.component.straggler_handling_functions.straggler_handling_function import (
    StragglerHandlingFunction,
)


class PercentageBasedStragglerHandling(StragglerHandlingFunction):
    def __init__(
        self,
        percent_collaborators_needed=1.0,
        minimum_reporting=1,
        **kwargs
    ):
        self.percent_collaborators_needed = percent_collaborators_needed
        self.minimum_reporting = minimum_reporting

    def minimum_collaborators_reported(self, num_collaborators_done):
        return num_collaborators_done >= self.minimum_reporting

    def straggler_cutoff_check(self, num_collaborators_done, all_collaborators):
        cutoff = (num_collaborators_done >= self.percent_collaborators_needed * len(
            all_collaborators)) and self.minimum_collaborators_reported(num_collaborators_done)
        return cutoff
