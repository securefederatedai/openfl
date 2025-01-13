# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils.tr_common_fixtures import (
    fx_federation_tr,
    fx_federation_tr_dws,
)
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)

# ** IMPORTANT **: This is just an example on how to add a test with below pre-requisites.
# Task Runner API Test function for federation run using sample_model
# 1. Create OpenFL workspace, if not present for the model and add relevant dataset and its path in plan/data.yaml
# 2. Append the model name to ModelName enum in tests/end_to_end/utils/constants.py
# 3. a. Use fx_federation_tr fixture for task runner with bare metal or docker approach.
# 3. b. Use fx_federation_tr_dws fixture for task runner with dockerized workspace approach.
# 4. Fixture will contain - model_owner, aggregator, collaborators, workspace_path, local_bind_path
# 5. Setup PKI for trusted communication within the federation based on TLS flag.
# 6. Start the federation using aggregator and given no of collaborators.
# 7. Verify the completion of the federation run.


@pytest.mark.task_runner_basic
def test_federation_basic(request, fx_federation_tr):
    """
    Add a proper docstring here.
    """
    log.info(f"Running sample model test {fx_federation_tr}")

    # Start the federation
    results = fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        results,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"


@pytest.mark.task_runner_dockerized_ws
def test_federation_via_dockerized_workspace(request, fx_federation_tr_dws):
    """
    Add a proper docstring here.
    """
    log.info(f"Running sample model test {fx_federation_tr_dws}")

    # Start the federation
    results = fed_helper.run_federation(
        fx_federation_tr_dws, use_tls=request.config.use_tls
    )

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_dws,
        results,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"
