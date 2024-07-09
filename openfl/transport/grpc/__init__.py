# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openfl.transport.grpc.aggregator_client import AggregatorGRPCClient
from openfl.transport.grpc.aggregator_server import AggregatorGRPCServer
from openfl.transport.grpc.director_server import DirectorGRPCServer

class ShardNotFoundError(Exception):
    """Indicates that director has no information about that shard."""
