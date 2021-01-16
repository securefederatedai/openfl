# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .federation_pb2 import ModelProto, MetadataProto
from .federation_pb2 import NamedTensor, DataStream
from .federation_pb2 import TasksRequest, TasksResponse, TaskResults
from .federation_pb2 import MessageHeader, Acknowledgement
from .federation_pb2 import TensorRequest, TensorResponse

from .federation_pb2_grpc import AggregatorServicer, add_AggregatorServicer_to_server
from .federation_pb2_grpc import AggregatorStub
