# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import os


from tests.end_to_end.utils.tr_common_fixtures import (
    fx_federation_tr,
)
from tests.end_to_end.utils import federation_helper as fed_helper
import json
import tests.end_to_end.utils.defaults as defaults

log = logging.getLogger(__name__)

# write a fixture to update request.config.num_rounds to 1
@pytest.fixture(scope="function")
def set_num_rounds(request):
    """
    Fixture to set the number of rounds for the test.
    Args:
        request (Fixture): Pytest fixture
    """
    # Set the number of rounds to 1
    log.info("Setting number of rounds to 1 for analytics test")
    request.config.num_rounds = 1
    if "federated_analytics" not in request.config.model_name:
        pytest.skip(
            f"Model name {request.config.model_name} is not supported for this test. "
            "Please use a different model name."
        )


@pytest.mark.task_runner_fed_analytics
def test_federation_analytics(request, set_num_rounds, fx_federation_tr):
    """
    Test federation via native task runner.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"

    # verify that results get saved in save/results.json
    result_path = os.path.join(
        fx_federation_tr.aggregator.workspace_path,
        "save",
        "result.json"
    )
    assert os.path.exists(result_path), f"Results file {result_path} does not exist"

    with open(result_path, "r") as f:
        results = f.read()
    try:
        json.loads(results)
    except json.JSONDecodeError as e:
        log.warning("Results file is not valid JSON. Raw content:\n%s", results)
        raise e

    assert results, f"Results file {result_path} is empty"
