# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Director tests module."""

from pathlib import Path
from unittest import mock

import pytest

from openfl.component.director import Director
from openfl.transport import DirectorGRPCServer


@pytest.fixture
def insecure_director():
    """Initialize an insecure director mock."""
    director = DirectorGRPCServer(director_cls=Director, tls=False)

    return director


@pytest.fixture
def secure_director():
    """Initialize a secure director mock."""
    director = DirectorGRPCServer(
        director_cls=Director,
        root_ca='./cert/root_ca.crt',
        key='./cert/localhost.key',
        cert='./cert/localhost.crt'
    )
    return director


def test_fill_certs(insecure_director, secure_director):
    """Test that fill_cert fill certificates params correctly."""
    assert insecure_director.root_ca is None
    assert insecure_director.key is None
    assert insecure_director.cert is None
    assert isinstance(secure_director.root_ca, Path)
    assert isinstance(secure_director.key, Path)
    assert isinstance(secure_director.cert, Path)
    with pytest.raises(Exception):
        secure_director.test_fill_certs('.', '.', None)
    with pytest.raises(Exception):
        secure_director.test_fill_certs('.', None, '.')
    with pytest.raises(Exception):
        secure_director.test_fill_certs(None, '.', '.')


def test_get_sender_tls(insecure_director):
    """Test that get_sender works correctly with TLS."""
    insecure_director.tls = True
    context = mock.Mock()
    sender = 'sender'
    context.auth_context = mock.Mock(
        return_value={'x509_common_name': [sender.encode('utf-8')]}
    )
    result = insecure_director.get_sender(context)
    assert result == sender


def test_get_sender_no_tls(insecure_director):
    """Test that get_sender works correctly without TLS."""
    context = mock.Mock()
    default_sender = 'unauthorized_sender'
    result = insecure_director.get_sender(context)
    assert result == default_sender
