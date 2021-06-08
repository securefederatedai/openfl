# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl.protocols module."""

from .federation_pb2 import Acknowledgement
from .federation_pb2 import DataStream
from .federation_pb2 import MessageHeader
from .federation_pb2 import MetadataProto
from .federation_pb2 import ModelProto
from .federation_pb2 import NamedTensor
from .federation_pb2 import TaskResults
from .federation_pb2 import TasksRequest
from .federation_pb2 import TasksResponse
from .federation_pb2 import TensorRequest
from .federation_pb2 import TensorResponse
from .federation_pb2_grpc import add_AggregatorServicer_to_server
from .federation_pb2_grpc import AggregatorServicer
from .federation_pb2_grpc import AggregatorStub
