# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenFL Component Module."""

from importlib import util

from openfl.component.aggregator.aggregator import Aggregator
from openfl.component.aggregator.straggler_handling import (
    CutoffTimePolicy,
    PercentagePolicy,
    StragglerPolicy,
    WaitForAllPolicy,
)
from openfl.component.assigner.assigner import Assigner
from openfl.component.assigner.random_grouped_assigner import RandomGroupedAssigner
from openfl.component.assigner.static_grouped_assigner import StaticGroupedAssigner
from openfl.component.collaborator.collaborator import Collaborator

if util.find_spec("flwr") is not None:
    from openfl.component.connector.connector_flower import ConnectorFlower
