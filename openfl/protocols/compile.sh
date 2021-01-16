# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
rm *_pb2.py *_pb2_grpc.py 2> /dev/null
python -m pip install -qq grpcio-tools
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./*.proto
sed -E 's/import (.+)_pb2 as/from . import \1_pb2 as/' *_pb2_grpc.py -i

