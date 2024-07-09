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

from openfl.component.aggregator.aggregator import Aggregator
from openfl.component.assigner.assigner import Assigner
from openfl.component.assigner.random_grouped_assigner import (
    RandomGroupedAssigner,
)
from openfl.component.assigner.static_grouped_assigner import (
    StaticGroupedAssigner,
)
from openfl.component.collaborator.collaborator import Collaborator
from openfl.component.straggler_handling_functions.cutoff_time_based_straggler_handling import (
    CutoffTimeBasedStragglerHandling,
)
from openfl.component.straggler_handling_functions.percentage_based_straggler_handling import (
    PercentageBasedStragglerHandling,
)
from openfl.component.straggler_handling_functions.straggler_handling_function import (
    StragglerHandlingFunction,
)
