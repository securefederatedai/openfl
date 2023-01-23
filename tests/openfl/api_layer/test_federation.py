# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Federation API tests module."""

from unittest import mock

import pytest

from openfl.interface.interactive_api.federation import Federation

CLIENT_ID = 'id test'
SAMPLE_SHAPE = (10, 10, 3)
TARGET_SHAPE = (2,)


@pytest.fixture
@mock.patch('openfl.interface.interactive_api.federation.DirectorClient')
def federation_object(mock_client_class):
    """Federation object fixture."""
    mock_client_instance = mock.Mock()
    mock_client_class.return_value = mock_client_instance
    mock_client_instance.get_dataset_info.return_value = (SAMPLE_SHAPE, TARGET_SHAPE)
    return Federation(client_id=CLIENT_ID)


def test_federation_initialization(federation_object):
    """Test Federation initialization."""
    assert federation_object.sample_shape == SAMPLE_SHAPE
    assert federation_object.target_shape == TARGET_SHAPE
    federation_object.dir_client.get_dataset_info.assert_called_once()


def test_dummy_shard_descriptor(federation_object):
    """Test dummy shard descriptor object."""
    dummy_shard_desc = federation_object.get_dummy_shard_descriptor(10)
    dummy_shard_dataset = dummy_shard_desc.get_dataset('')
    sample, target = dummy_shard_dataset[0]
    assert sample.shape == SAMPLE_SHAPE
    assert target.shape == TARGET_SHAPE
