# Copyright (C) 2020-2023 Intel Corporation
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
        root_certificate=Path('./cert/root_ca.crt').absolute(),
        private_key=Path('./cert/localhost.key').absolute(),
        certificate=Path('./cert/localhost.crt').absolute()
    )
    return director


def test_fill_certs(insecure_director, secure_director):
    """Test that fill_cert fill certificates params correctly."""
    assert insecure_director.root_certificate is None
    assert insecure_director.private_key is None
    assert insecure_director.certificate is None
    assert isinstance(secure_director.root_certificate, Path)
    assert isinstance(secure_director.private_key, Path)
    assert isinstance(secure_director.certificate, Path)
    with pytest.raises(Exception):
        secure_director._fill_certs('.', '.', None)
    with pytest.raises(Exception):
        secure_director._fill_certs('.', None, '.')
    with pytest.raises(Exception):
        secure_director._fill_certs(None, '.', '.')
    secure_director._fill_certs('.', '.', '.')


def test_get_caller_tls(insecure_director):
    """Test that get_caller works correctly with TLS."""
    insecure_director.tls = True
    context = mock.Mock()
    client_id = 'client_id'
    context.auth_context = mock.Mock(
        return_value={'x509_common_name': [client_id.encode('utf-8')]}
    )
    result = insecure_director.get_caller(context)
    assert result == client_id


def test_get_sender_no_tls(insecure_director):
    """Test that get_sender works correctly without TLS."""
    context = mock.Mock()
    client_id = 'client_id'
    context.invocation_metadata.return_value = (('client_id', client_id),)
    result = insecure_director.get_caller(context)
    assert result == client_id


def test_get_sender_no_tls_no_client_id(insecure_director):
    """Test that get_sender works correctly without TLS and client_id."""
    context = mock.Mock()
    context.invocation_metadata = mock.Mock()
    context.invocation_metadata.return_value = (('key', 'value'),)
    default_client_id = '__default__'
    result = insecure_director.get_caller(context)
    assert result == default_client_id
