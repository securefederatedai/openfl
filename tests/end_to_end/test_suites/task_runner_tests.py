# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils.common_fixtures import fx_federation_tr
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


# NOTE: This test file contains the test cases for the task runner federation using bare metal and docker approaches.
# Fixture and marker mapping:
# fx_federation_tr - task_runner_basic and task_runner_docker
# fx_federation_tr_dws - task_runner_dockerized_ws

@pytest.mark.task_runner_basic
def test_federation_via_native(request, fx_federation_tr):
    """
    Test federation via native task runner.
    """
    # Start the federation
    results = fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr, results, num_rounds=request.config.num_rounds
    ), "Federation completion failed"


@pytest.mark.task_runner_docker
def test_federation_via_docker(request, fx_federation_tr):
    """
    Test federation via docker.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture
    """
    # Start the federation
    results = fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr, results, request.config.num_rounds
    ), "Federation completion failed"
