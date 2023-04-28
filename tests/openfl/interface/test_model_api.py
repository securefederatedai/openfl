# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Model interface tests module."""

from unittest import mock
from unittest import TestCase
from pathlib import Path

from openfl import get_model
from openfl.interface.model import save_
from openfl.federated.task import TaskRunner


@mock.patch('openfl.interface.model.utils')
@mock.patch('openfl.interface.model.Plan')
def test_get_model(Plan, utils):  # noqa: N803
    "Test get_module returns TaskRunner."
    plan_instance = mock.Mock()
    plan_instance.cols_data_paths = ['mock_col_name']
    Plan.parse.return_value = plan_instance

    plan_instance.get_task_runner.return_value = TaskRunner(data_loader=mock.Mock())
    TaskRunner.set_tensor_dict = mock.Mock()

    tensor_dict = mock.Mock()
    utils.deconstruct_model_proto.return_value = tensor_dict, {}

    # Function call
    result = get_model('plan_path', 'cols_path', 'data_path', 'model_protobuf_path')

    # Asserts below
    Plan.parse.assert_called_once()

    utils.load_proto.assert_called_once()
    utils.deconstruct_model_proto.assert_called_once()

    plan_instance.get_task_runner.assert_called_once()

    TaskRunner.set_tensor_dict.assert_called_once_with(tensor_dict, with_opt_vars=False)

    assert isinstance(result, TaskRunner)


@mock.patch('openfl.interface.model.get_model')
def test_model_save(mock_get_model):
    current_path = Path(__file__).resolve()
    plan_path = current_path.parent.joinpath('plan')
    plan_config = plan_path.joinpath('plan.yaml')
    cols_config = plan_path.joinpath('cols.yaml')
    data_config = plan_path.joinpath('data.yaml')

    mock_get_model.return_value = TaskRunner(data_loader=mock.Mock())

    with TestCase.assertRaises(test_model_save, NotImplementedError):
        save_(['-i', current_path,
               '-p', plan_config,
               '-c', cols_config,
               '-d', data_config], standalone_mode=False)

    mock_get_model.assert_called_once()
