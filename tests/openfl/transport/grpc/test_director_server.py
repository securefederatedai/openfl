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


@pytest.mark.parametrize('caller_common_name,caller_cert_name,result,tls', [
    ('one', 'one', True, True), ('one', 'two', False, True),
    ('one', 'one', True, False), ('one', 'two', True, False)])
def test_validate_caller(insecure_director, caller_common_name,
                         caller_cert_name, result, tls):
    """Test that validate_caller works correctly."""
    insecure_director.tls = tls  # only for enable TLS the validation works
    request = mock.Mock()
    request.header = mock.Mock()
    request.header.sender = caller_common_name
    context = mock.Mock()
    context.auth_context = mock.Mock(
        return_value={'x509_common_name': [caller_cert_name.encode('utf-8')]}
    )

    assert result == insecure_director.validate_caller(request, context)
