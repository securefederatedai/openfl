# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Plan API's tests module."""

from unittest import mock

import pytest
from pathlib import Path

from openfl.federated.plan.plan import Plan
from openfl.component.assigner import RandomGroupedAssigner
from openfl.component.aggregator import Aggregator
from openfl.transport.rest.aggregator_server import AggregatorRESTServer
from openfl.transport.grpc.aggregator_server import AggregatorGRPCServer


@pytest.fixture
def plan():
    return Plan.parse(Path(__file__).parent / 'plan_example.yaml', resolve=True)


@pytest.fixture
def empty_plan():
    return Plan()


def test_import():
    assert isinstance(Plan.import_('openfl.federated.plan.plan.Plan'), type(Plan))


def test_build(plan):
    defaults = plan.config['assigner']
    defaults['settings']['authorized_cols'] = ['col1']
    defaults['settings']['rounds_to_train'] = 1
    defaults['settings']['tasks'] = plan.get_tasks()
    assert isinstance(Plan.build(plan.config['assigner']['template'],
                                 plan.config['assigner']['settings']), RandomGroupedAssigner)


def test_get_assigner(plan):
    assert isinstance(plan.get_assigner(), RandomGroupedAssigner)


def test_get_tasks(empty_plan):
    assert isinstance(empty_plan.get_tasks(), dict)


def test_get_aggregator(mocker, plan):
    mocker.patch('openfl.protocols.utils.load_proto', mock.Mock())
    Aggregator._load_initial_tensors = mock.Mock()
    assert isinstance(plan.get_aggregator(), Aggregator)

def test_get_server_rest(plan,mocker):
    mocker.patch('openfl.protocols.utils.load_proto', return_value=mock.Mock())
    mock_setup_ssl = mocker.patch('openfl.transport.rest.aggregator_server.AggregatorRESTServer._setup_ssl_context', return_value=mock.Mock())
    plan.config['network']['settings']['transport_protocol'] = 'rest'
    server = plan.get_server()
    assert isinstance(server, AggregatorRESTServer)

def test_get_server_grpc(plan,mocker):
    mocker.patch('openfl.protocols.utils.load_proto', return_value=mock.Mock())
    plan.config['network']['settings']['transport_protocol'] = 'grpc'
    server = plan.get_server()
    assert isinstance(server, AggregatorGRPCServer)

def test_get_server_default_certificates(plan,mocker):
    mocker.patch('openfl.protocols.utils.load_proto', return_value=mock.Mock())
    server = plan.get_server()
    assert isinstance(server, AggregatorGRPCServer)  # Default to gRPC

def test_get_server_invalid_protocol(plan,mocker):
    mocker.patch('openfl.protocols.utils.load_proto', return_value=mock.Mock())
    plan.config['network']['settings']['transport_protocol'] = 'invalid_protocol'
    with pytest.raises(ValueError):
        plan.get_server()
