# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Collaborator interface tests module."""

from unittest import mock
from unittest import TestCase
from pathlib import Path

from openfl.interface.collaborator import start_, register_data_path


@mock.patch('openfl.federated.Plan.parse')
def test_collaborator_start(mock_parse):
    current_path = Path(__file__).resolve()
    plan_path = current_path.parent.joinpath('plan')
    plan_config = plan_path.joinpath('plan.yaml')
    data_config = plan_path.joinpath('data.yaml')

    mock_parse.return_value = mock.Mock()

    ret = start_(['-p', plan_config,
                  '-d', data_config,
                  '-n', 'one'], standalone_mode=False)
    assert ret is None


@mock.patch('openfl.interface.collaborator.is_directory_traversal')
@mock.patch('openfl.federated.Plan.parse')
def test_collaborator_start_illegal_plan(mock_parse, mock_is_directory_traversal):
    current_path = Path(__file__).resolve()
    plan_path = current_path.parent.joinpath('plan')
    plan_config = plan_path.joinpath('plan.yaml')
    data_config = plan_path.joinpath('data.yaml')

    mock_parse.return_value = mock.Mock()
    mock_is_directory_traversal.side_effect = [True, False]

    with TestCase.assertRaises(test_collaborator_start_illegal_plan, SystemExit):
        start_(['-p', plan_config,
                '-d', data_config,
                '-n', 'one'], standalone_mode=False)


@mock.patch('openfl.interface.collaborator.is_directory_traversal')
@mock.patch('openfl.federated.Plan.parse')
def test_collaborator_start_illegal_data(mock_parse, mock_is_directory_traversal):
    current_path = Path(__file__).resolve()
    plan_path = current_path.parent.joinpath('plan')
    plan_config = plan_path.joinpath('plan.yaml')
    data_config = plan_path.joinpath('data.yaml')

    mock_parse.return_value = mock.Mock()
    mock_is_directory_traversal.side_effect = [False, True]

    with TestCase.assertRaises(test_collaborator_start_illegal_plan, SystemExit):
        start_(['-p', plan_config,
                '-d', data_config,
                '-n', 'one'], standalone_mode=False)


@mock.patch('genericpath.isfile')
def test_collaborator_register_data_path(mock_isfile):
    mock_isfile.return_value = True
    ret = register_data_path('one', 'path/data')
    assert ret is None
