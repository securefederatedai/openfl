# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Derector Envoy's client tests module."""

from unittest import mock

import pytest

from openfl.transport.grpc.director_client import ShardDirectorClient


@pytest.fixture
@mock.patch('openfl.transport.grpc.director_client.director_pb2_grpc')
def director_client(director_pb2_grpc):
    """Director client fixture."""
    director_pb2_grpc.FederationDirectorStub.return_value = mock.Mock()

    director_host = 'fqdn'
    director_port = 50051
    shard_name = 'test shard'
    tls = False
    root_certificate, private_key, certificate = None, None, None
    director_client = ShardDirectorClient(
        director_host=director_host,
        director_port=director_port,
        shard_name=shard_name,
        tls=tls,
        root_certificate=root_certificate,
        private_key=private_key,
        certificate=certificate,
    )
    return director_client


def test_report_shard_info(director_client):
    """Test report_shard_info RPC."""
    shard_descriptor = mock.MagicMock()
    shard_descriptor.dataset_description = 'description'
    shard_descriptor.__len__.return_value = 10
    shard_descriptor.sample_shape = [str(dim) for dim in (1, 2)]
    shard_descriptor.target_shape = [str(dim) for dim in (10,)]

    director_client.report_shard_info(shard_descriptor)

    director_client.stub.AcknowledgeShard.assert_called_once()
    shard_info = director_client.stub.AcknowledgeShard.call_args.args[0]
    assert shard_info.shard_description == shard_descriptor.dataset_description
    assert shard_info.n_samples == 10
    assert shard_info.sample_shape == shard_descriptor.sample_shape
