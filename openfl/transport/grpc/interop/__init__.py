# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import util

if util.find_spec("flwr") is not None:
    from openfl.transport.grpc.interop.flower.interop_client import FlowerInteropClient
    from openfl.transport.grpc.interop.flower.interop_server import FlowerInteropServer
