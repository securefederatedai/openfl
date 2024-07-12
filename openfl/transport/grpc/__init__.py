# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.transport.grpc package."""

from .aggregator_client import AggregatorGRPCClient
from .aggregator_server import AggregatorGRPCServer
from .director_server import DirectorGRPCServer


class ShardNotFoundError(Exception):
    """Indicates that director has no information about that shard."""


__all__ = [
    'AggregatorGRPCServer',
    'AggregatorGRPCClient',
    'DirectorGRPCServer',
    'ShardNotFoundError',
]
