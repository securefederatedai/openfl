# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenFL Connector Module."""

from importlib import util

if util.find_spec("flwr") is not None:
    from openfl.component.connector.connector_flower import ConnectorFlower
