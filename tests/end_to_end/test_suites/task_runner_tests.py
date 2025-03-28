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


@pytest.mark.task_runner_basic
def test_federation_via_native(request, fx_federation_tr):
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

    best_agg_score = fed_helper.get_best_agg_score(fx_federation_tr.aggregator.tensor_db_file)
    log.info(f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}")


@pytest.mark.task_runner_dockerized_ws
def test_federation_via_dockerized_workspace(request, fx_federation_tr_dws):
    """
    Test federation via dockerized workspace.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr_dws (Fixture): Pytest fixture for dockerized workspace
    """
    # Start the federation
    assert fed_helper.run_federation_for_dws(fx_federation_tr_dws, request.config.use_tls)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_dws,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"

    best_agg_score = fed_helper.get_best_agg_score(fx_federation_tr_dws.aggregator.tensor_db_file)
    log.info(f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}")
