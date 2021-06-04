# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.transport.grpc package."""

from .client import CollaboratorGRPCClient
from .server import AggregatorGRPCServer

__all__ = [
    'AggregatorGRPCServer',
    'CollaboratorGRPCClient',
]
