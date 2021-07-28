import pytest
from unittest import mock

from openfl.interface.interactive_api.federation import Federation

CLIENT_ID = 'id test'
SAMPLE_SHAPE = (10, 10, 3)
TARGET_SHAPE = (2,)


@pytest.fixture
@mock.patch('openfl.interface.interactive_api.federation.DirectorClient')
def federation_object(mock_client_class):
    mock_client_instance = mock.Mock()
    mock_client_class.return_value = mock_client_instance
    mock_client_instance.get_dataset_info.return_value = (SAMPLE_SHAPE, TARGET_SHAPE)
    return Federation(client_id=CLIENT_ID)


def test_federation_initialization(federation_object):
    assert federation_object.sample_shape == SAMPLE_SHAPE
    assert federation_object.target_shape == TARGET_SHAPE
    federation_object.dir_client.get_dataset_info.assert_called_once()


def test_dummy_shard_descriptor(federation_object):
    dummy_sd = federation_object.get_dummy_shard_descriptor(10)
    assert len(dummy_sd) == 10
    sample, target = dummy_sd[0]
    assert sample.shape == SAMPLE_SHAPE
    assert target.shape == TARGET_SHAPE
