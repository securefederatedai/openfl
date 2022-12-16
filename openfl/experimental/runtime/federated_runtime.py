# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime module FederatedRuntime class."""

from openfl.experimental.runtime import Runtime


class FederatedRuntime(Runtime):
    def __init__(self, aggregator, collaborators=None):
        """Use remote federated infrastructure to run the flow"""
        raise NotImplementedError("FederatedRuntime will be implemented in the future")
