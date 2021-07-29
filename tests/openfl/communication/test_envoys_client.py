import pytest
from unittest import mock

from openfl.protocols import director_pb2
from openfl.transport.grpc.director_client import ShardDirectorClient


@pytest.fixture
@mock.patch('openfl.transport.grpc.director_client.director_pb2_grpc')
def director_client(director_pb2_grpc):
    director_pb2_grpc.FederationDirectorStub.return_value = mock.Mock()

    director_uri = 'fqdn'
    shard_name = 'test shard'
    disable_tls = True
    root_ca, key, cert = None, None, None
    director_client = ShardDirectorClient(
        director_uri, shard_name, disable_tls,
        root_ca, key, cert)
    return director_client


def test_repotr_shard_info(director_client):
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
