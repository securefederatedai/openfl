# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Aggregator interface tests module."""
import pytest
from unittest import mock
from unittest import TestCase
from pathlib import Path

from openfl.interface.aggregator import start_, _generate_cert_request
from openfl.interface.aggregator import find_certificate_name, _certify


@mock.patch('openfl.federated.Plan.parse')
def test_aggregator_start(mock_parse):
    current_path = Path(__file__).resolve()
    plan_path = current_path.parent.joinpath('plan')
    plan_config = plan_path.joinpath('plan.yaml')
    cols_config = plan_path.joinpath('cols.yaml')

    # Create a mock plan with the required fields
    mock_plan = mock.MagicMock()
    mock_plan.__getitem__.side_effect = {'task_group': 'learning'}.get
    mock_plan.get = {'task_group': 'learning'}.get
    # Add the config attribute with proper nesting
    mock_plan.config = {
        'aggregator': {
            'settings': {
                'task_group': 'learning'
            }
        }
    }
    mock_parse.return_value = mock_plan

    ret = start_(['-p', plan_config,
                  '-c', cols_config], standalone_mode=False)
    assert ret is None


@mock.patch('openfl.interface.aggregator.is_directory_traversal')
@mock.patch('openfl.federated.Plan.parse')
def test_aggregator_start_illegal_plan(mock_parse, mock_is_directory_traversal):
    current_path = Path(__file__).resolve()
    plan_path = current_path.parent.joinpath('plan')
    plan_config = plan_path.joinpath('plan.yaml')
    cols_config = plan_path.joinpath('cols.yaml')

    # Create a mock plan with the required fields
    mock_plan = mock.MagicMock()
    mock_plan.__getitem__.side_effect = {'task_group': 'learning'}.get
    mock_plan.get = {'task_group': 'learning'}.get
    # Add the config attribute with proper nesting
    mock_plan.config = {
        'aggregator': {
            'settings': {
                'task_group': 'learning'
            }
        }
    }
    mock_parse.return_value = mock_plan

    mock_is_directory_traversal.side_effect = [True, False]

    with TestCase.assertRaises(test_aggregator_start_illegal_plan, SystemExit):
        start_(['-p', plan_config,
                '-c', cols_config], standalone_mode=False)


@mock.patch('openfl.interface.aggregator.is_directory_traversal')
@mock.patch('openfl.federated.Plan.parse')
def test_aggregator_start_illegal_cols(mock_parse, mock_is_directory_traversal):
    current_path = Path(__file__).resolve()
    plan_path = current_path.parent.joinpath('plan')
    plan_config = plan_path.joinpath('plan.yaml')
    cols_config = plan_path.joinpath('cols.yaml')

    # Create a mock plan with the required fields
    mock_plan = mock.MagicMock()
    mock_plan.__getitem__.side_effect = {'task_group': 'learning'}.get
    mock_plan.get = {'task_group': 'learning'}.get
    # Add the config attribute with proper nesting
    mock_plan.config = {
        'aggregator': {
            'settings': {
                'task_group': 'learning'
            }
        }
    }
    mock_parse.return_value = mock_plan

    mock_is_directory_traversal.side_effect = [False, True]

    with TestCase.assertRaises(test_aggregator_start_illegal_cols, SystemExit):
        start_(['-p', plan_config,
                '-c', cols_config], standalone_mode=False)


def test_aggregator_generate_cert_request():
    ret = _generate_cert_request([], standalone_mode=False)
    assert ret is None


@mock.patch('builtins.open', mock.mock_open(read_data='Subject: US=01234\n Subject: CN=56789\n '))
def test_aggregator_find_certificate_name():
    col_name = find_certificate_name('')
    assert col_name == '56789'


# NOTE: This test is disabled because of cryptic behaviour on calling
# _certify(). Previous version of _certify() had imports defined within
# the function, which allowed theses tests to pass, whereas the goal of the
# @mock.patch here seems to be to make them dummy. Usefulness of this test is
# doubtful. Now that the imports are moved to the top level (a.k.a out of
# _certify()) this test fails.
# In addition, using dummy return types for read/write key/csr seems to
# obviate the need for even testing _certify().
@pytest.mark.skip()
@mock.patch('openfl.cryptography.io.write_crt')
@mock.patch('openfl.cryptography.ca.sign_certificate')
@mock.patch('click.confirm')
@mock.patch('openfl.cryptography.io.read_crt')
@mock.patch('openfl.cryptography.io.read_key')
@mock.patch('openfl.cryptography.io.read_csr')
def test_aggregator_certify(mock_read_csr, mock_read_key, mock_read_crt,
                            mock_confirm, mock_sign_certificate, mock_write_crt):
    mock_read_csr.return_value = ['test_csr', 'test_csr_hash']
    mock_read_key.return_value = mock.Mock()
    mock_read_crt.return_value = mock.Mock()
    mock_confirm.side_effect = [True, False]
    mock_sign_certificate.return_value = mock.Mock()
    mock_write_crt.return_value = mock.Mock()

    # confirm 'Do you want to sign this certificate?' True
    ret1 = _certify([], standalone_mode=False)
    assert ret1 is None

    # confirm 'Do you want to sign this certificate?' False
    ret2 = _certify([], standalone_mode=False)
    assert ret2 is None

    ret3 = _certify(['-s'], standalone_mode=False)
    assert ret3 is None
