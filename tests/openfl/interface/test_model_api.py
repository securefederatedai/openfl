# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Model interface tests module."""

from unittest import mock
from unittest import TestCase
from pathlib import Path

from openfl.interface.model import save_
from openfl.federated.task import TaskRunner


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
