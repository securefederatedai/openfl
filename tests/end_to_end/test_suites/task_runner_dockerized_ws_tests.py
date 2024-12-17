# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import concurrent.futures

import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.utils.common_fixtures import fx_federation_tr_dws
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


# NOTE: This test file contains the test cases for the task runner federation using dockerized workspace approach.
# Fixture and marker mapping:
# fx_federation_tr - task_runner_basic and task_runner_docker
# fx_federation_tr_dws - task_runner_dockerized_ws

@pytest.mark.task_runner_dockerized_ws
def test_federation_via_dockerized_workspace(request, fx_federation_tr_dws):
    """
    Test federation via dockerized workspace.
    Args:
        request (Fixture): Pytest fixture
        fx_federation (Fixture): Pytest fixture
    """
    # Start the federation
    results = fed_helper.run_federation_for_dws(fx_federation_tr_dws, use_tls=request.config.use_tls)

    log.info(f"Federation run completed successfully with {results}")
    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation_tr_dws, results, request.config.num_rounds), "Federation completion failed"
