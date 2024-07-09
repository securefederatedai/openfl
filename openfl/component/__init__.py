# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openfl.component.aggregator import Aggregator
from openfl.component.assigner import Assigner, RandomGroupedAssigner, StaticGroupedAssigner
from openfl.component.collaborator import Collaborator
from openfl.component.straggler_handling_functions import (
    CutoffTimeBasedStragglerHandling,
    PercentageBasedStragglerHandling,
    StragglerHandlingFunction,
)