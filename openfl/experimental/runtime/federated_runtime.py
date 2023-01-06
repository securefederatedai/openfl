# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime module FederatedRuntime class."""

from __future__ import annotations
from openfl.experimental.runtime import Runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator, FLSpec


class FederatedRuntime(Runtime):
    def __init__(self, aggregator: Type[Aggregator], collaborators: List[Type[Collaborator]] = None):
        """Use remote federated infrastructure to run the flow"""
        raise NotImplementedError("FederatedRuntime will be implemented in the future")
