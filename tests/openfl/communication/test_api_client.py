# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Derector API's client tests module."""

import sys
from unittest import mock

import pytest

from openfl.protocols import director_pb2
from openfl.transport.grpc.director_client import DirectorClient


@pytest.fixture
@mock.patch('openfl.transport.grpc.director_client.director_pb2_grpc')
def director_client(director_pb2_grpc):
    """Director client fixture."""
    director_pb2_grpc.DirectorStub.return_value = mock.Mock()

    client_id = 'one'
    director_host = 'localhost'
    director_port = 50051
    tls = False
    root_certificate, private_key, certificate = None, None, None
    director_client = DirectorClient(
        director_host=director_host,
        director_port=director_port,
        client_id=client_id,
        tls=tls,
        root_certificate=root_certificate,
        private_key=private_key,
        certificate=certificate
    )
    return director_client


def test_get_dataset_info(director_client):
    """Test get_dataset_info RPC."""
    director_client.get_dataset_info()
    director_client.stub.GetDatasetInfo.assert_called_once()


@pytest.mark.parametrize(
    'clients_method,model_type', [
        ('get_best_model', 'BEST_MODEL'),
        ('get_last_model', 'LAST_MODEL'),
    ])
@mock.patch('openfl.transport.grpc.director_client.deconstruct_model_proto')
def test_get_best_model(deconstruct_model_proto, director_client,
                        clients_method, model_type):
    """Test get_best_model RPC."""
    deconstruct_model_proto.return_value = {}, {}
    getattr(director_client, clients_method)('test name')
    director_client.stub.GetTrainedModel.assert_called_once()

    request = director_client.stub.GetTrainedModel.call_args
    if sys.version_info < (3, 8):
        incoming_model_type = request[0][0].model_type
    else:
        incoming_model_type = request.args[0].model_type
    assert incoming_model_type == getattr(director_pb2.GetTrainedModelRequest, model_type)
