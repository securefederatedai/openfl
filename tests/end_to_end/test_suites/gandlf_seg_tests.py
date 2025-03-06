# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils import federation_helper as fed_helper
from tests.end_to_end.utils.tr_workspace import create_tr_workspace_gandlf

log = logging.getLogger(__name__)


def test_gandlf_segmentation(request, fx_federation_tr_gandlf):
    """
    Test federation via native task runner with GaNDLF.
    IMPORTANT - ensure that all the pre-requisites steps for GanDLF are met before this test is run.
    Refer file .github/workflows/gandlf.yaml for the same.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr_gandlf)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_gandlf,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"

    best_agg_score = fed_helper.get_best_agg_score(fx_federation_tr_gandlf.aggregator.tensor_db_file)
    log.info(f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}")


@pytest.fixture(scope="function")
def fx_federation_tr_gandlf(request):
    """
    Fixture for federation in case of GANDLF model. This fixture is used to create the model owner, aggregator, and collaborators.
    It also creates workspace.
    Assumption: OpenFL workspace is present for the model being tested.
    Args:
        request: pytest request object. Model name is passed as a parameter to the fixture from test cases.
    Returns:
        federation_details: Named tuple containing the objects for model owner, aggregator, and collaborators

    Note: As this is a function level fixture, thus no import is required at test level.
    """
    request.config.test_env = "task_runner_basic_gandlf"
    return create_tr_workspace_gandlf(request)
