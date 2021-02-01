# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.transport.grpc package."""

from .server import AggregatorGRPCServer
from .client import CollaboratorGRPCClient


__all__ = [
    'AggregatorGRPCServer',
    'CollaboratorGRPCClient',
]
