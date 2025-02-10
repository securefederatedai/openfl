# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import logging

from tests.end_to_end.utils.tr_common_fixtures import (
    fx_federation_tr,
    fx_federation_tr_dws,
)
from tests.end_to_end.utils import federation_helper as fed_helper
from tests.end_to_end.utils.tr_workspace import create_tr_workspace, create_tr_dws_workspace

log = logging.getLogger(__name__)

TOLERANCE = 0.00001

@pytest.mark.task_runner_basic
def test_eval_federation_via_native(request, fx_federation_tr):
    """
    Test learning and evaluation steps via native task runner.
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

    # Set the best model path in request. It is used during plan initialization for evaluation step
    request.config.best_model_path = os.path.join(fx_federation_tr.aggregator.workspace_path, "save", "best.pbuf")
    best_model_score = fed_helper.get_best_agg_score(fx_federation_tr.aggregator.tensor_db_file)
    log.info(f"Model score post {request.config.num_rounds} rounds: {best_model_score}")
    # Create new workspace with evaluation scope
    new_fed_obj = create_tr_workspace(request, eval_scope=True)

    assert fed_helper.run_federation(new_fed_obj)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        new_fed_obj,
        test_env=request.config.test_env,
        num_rounds=1,
    ), "Federation completion failed"

    best_model_score_eval = fed_helper.get_best_agg_score(new_fed_obj.aggregator.tensor_db_file)
    log.info(f"Model score post {request.config.num_rounds} rounds: {best_model_score}")

    # verify that the model score is similar to the previous model score max of 0.001% difference
    assert abs(best_model_score - best_model_score_eval) <= TOLERANCE, "Model score is not similar to the previous score"


@pytest.mark.task_runner_dockerized_ws
def test_eval_federation_via_dockerized_workspace(request, fx_federation_tr_dws):
    """
    Test learning and evaluation steps via dockerized workspace.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr_dws (Fixture): Pytest fixture for dockerized workspace
    """
    # Start the federation
    assert fed_helper.run_federation_for_dws(
        fx_federation_tr_dws, use_tls=request.config.use_tls
    )

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_dws,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"

    # Set the best model path in request. It is used during plan initialization for evaluation step
    request.config.best_model_path = os.path.join(fx_federation_tr_dws.aggregator.workspace_path, "save", "best.pbuf")

    best_model_score = fed_helper.get_best_agg_score(fx_federation_tr_dws.aggregator.tensor_db_file)
    log.info(f"Model score post {request.config.num_rounds} rounds: {best_model_score}")

    # Create new workspace with evaluation scope
    new_fed_obj = create_tr_dws_workspace(request, eval_scope=True)

    assert fed_helper.run_federation_for_dws(new_fed_obj, use_tls=request.config.use_tls)
    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        new_fed_obj,
        test_env=request.config.test_env,
        num_rounds=1,
    ), "Federation completion failed"

    best_model_score_eval = fed_helper.get_best_agg_score(new_fed_obj.aggregator.tensor_db_file)
    log.info(f"Model score post {request.config.num_rounds} rounds: {best_model_score}")

    # verify that the model score is similar to the previous model score max of 0.001% difference
    assert abs(best_model_score - best_model_score_eval) <= TOLERANCE, "Model score is not similar to the previous score"
