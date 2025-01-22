# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pass
pass
import logging

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.federation_helper as fh
import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.models import aggregator as agg_model, model_owner as mo_model
from tests.end_to_end.utils.tr_workspace import create_tr_workspace, create_tr_workspace_dws

log = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def fx_federation_tr(request):
    """
    Fixture for federation. This fixture is used to create the model owner, aggregator, and collaborators.
    It also creates workspace.
    Assumption: OpenFL workspace is present for the model being tested.
    Args:
        request: pytest request object. Model name is passed as a parameter to the fixture from test cases.
    Returns:
        federation_details: Named tuple containing the objects for model owner, aggregator, and collaborators

    Note: As this is a function level fixture, thus no import is required at test level.
    """
    request.config.test_env = "task_runner_basic"
    return create_tr_workspace(request)


@pytest.fixture(scope="function")
def fx_federation_tr_dws(request):
    """
    Fixture for federation in case of dockerized workspace. This fixture is used to create the model owner, aggregator, and collaborators.
    It also creates workspace.
    Assumption: OpenFL workspace is present for the model being tested.
    Args:
        request: pytest request object. Model name is passed as a parameter to the fixture from test cases.
    Returns:
        federation_details: Named tuple containing the objects for model owner, aggregator, and collaborators

    Note: As this is a function level fixture, thus no import is required at test level.
    """
    request.config.test_env = "task_runner_dockerized_ws"
    return create_tr_workspace_dws(request)
