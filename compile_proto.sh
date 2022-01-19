# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
rm openfl/protocols/*_pb2.py openfl/protocols/*_pb2_grpc.py 2> /dev/null
python -m pip install -qq grpcio-tools
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. \
  openfl/protocols/aggregator.proto \
  openfl/protocols/director.proto
python -m grpc_tools.protoc -I . --python_out=. openfl/protocols/base.proto