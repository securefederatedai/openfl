# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.transport.grpc package."""

from .aggregator_client import AggregatorGRPCClient
from .aggregator_server import AggregatorGRPCServer


class ShardNotFoundError(Exception):
    """Indicates that director has no information about that shard."""


__all__ = [
    'AggregatorGRPCServer',
    'AggregatorGRPCClient',
    'ShardNotFoundError',
]
