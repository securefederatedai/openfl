# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Experiment representation class tests module."""

from pathlib import Path
from unittest import mock

import pytest

from openfl.component.director.experiment import Experiment


@pytest.fixture
@mock.patch('openfl.component.director.experiment.Path')
def experiment_rep(_):
    """Initialize an experiment."""
    experiment = Experiment(
        name='test_exp',
        archive_path=mock.MagicMock(spec_set=Path),
        collaborators=['test_col'],
        sender='test_user',
        init_tensor_dict={},
    )
    return experiment


@pytest.mark.asyncio
@mock.patch('openfl.component.director.experiment.ExperimentWorkspace')
@pytest.mark.parametrize('review_result,archive_unlink_call_count,experiment_status',
                         [(True, 0, 'pending'), (False, 1, 'rejected')]
                         )
async def test_review_experiment(
    ExperimentWorkspace, experiment_rep, review_result,
    archive_unlink_call_count, experiment_status
):
    """Review experiment method test."""
    review_callback = mock.Mock()
    review_callback.return_value = review_result

    result = await experiment_rep.review_experiment(review_plan_callback=review_callback)

    assert result is review_callback.return_value
    ExperimentWorkspace.assert_called_once_with(
        'test_exp',
        experiment_rep.archive_path,
        is_install_requirements=False,
        remove_archive=False
    )
    assert experiment_rep.archive_path.unlink.call_count == archive_unlink_call_count
    assert experiment_rep.status == experiment_status
