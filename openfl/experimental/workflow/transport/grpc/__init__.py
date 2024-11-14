# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.transport.grpc package."""

from openfl.experimental.workflow.transport.grpc.aggregator_client import AggregatorGRPCClient
from openfl.experimental.workflow.transport.grpc.aggregator_server import AggregatorGRPCServer


# FIXME: Not the right place for exceptions
class ShardNotFoundError(Exception):
    """Indicates that director has no information about that shard."""
