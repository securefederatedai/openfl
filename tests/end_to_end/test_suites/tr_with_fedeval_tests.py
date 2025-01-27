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
from tests.end_to_end.utils.summary_helper import get_aggregated_accuracy

log = logging.getLogger(__name__)


@pytest.mark.task_runner_basic
def test_eval_federation_via_native(request, fx_federation_tr):
    """
    Test learning and evaluation steps via native task runner.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    results = fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        results,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"

    # Set the best model path in request. It is used during plan initialization for evaluation step
    request.config.best_model_path = os.path.join(fx_federation_tr.aggregator.workspace_path, "save", "best.pbuf")
    metric_file_path = os.path.join(fx_federation_tr.aggregator.workspace_path, "logs", "aggregator_metrics.txt")
    model_accuracy = get_aggregated_accuracy(metric_file_path)
    log.info(f"Model accuracy post {request.config.num_rounds} rounds: {model_accuracy}")
    # Create new workspace with evaluation scope
    new_fed_obj = create_tr_workspace(request, eval_scope=True)

    results_new = fed_helper.run_federation(new_fed_obj)
    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        new_fed_obj,
        results_new,
        test_env=request.config.test_env,
        num_rounds=1,
    ), "Federation completion failed"

    new_metric_file_path = os.path.join(new_fed_obj.aggregator.workspace_path, "logs", "aggregator_metrics.txt")
    model_accuracy_eval = get_aggregated_accuracy(new_metric_file_path)
    log.info(f"Model accuracy during evaluation only on prev trained model is : {model_accuracy_eval})")

    # verify that the model accuracy is similar to the previous model accuracy max of 1% difference
    assert abs(model_accuracy - model_accuracy_eval) <= 0.01, "Model accuracy is not similar to the previous model accuracy"


@pytest.mark.task_runner_dockerized_ws
def test_eval_federation_via_dockerized_workspace(request, fx_federation_tr_dws):
    """
    Test learning and evaluation steps via dockerized workspace.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr_dws (Fixture): Pytest fixture for dockerized workspace
    """
    # Start the federation
    results = fed_helper.run_federation_for_dws(
        fx_federation_tr_dws, use_tls=request.config.use_tls
    )

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_dws,
        results,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"

    # Set the best model path in request. It is used during plan initialization for evaluation step
    request.config.best_model_path = os.path.join(fx_federation_tr_dws.aggregator.workspace_path, "save", "best.pbuf")
    metric_file_path = os.path.join(fx_federation_tr_dws.aggregator.workspace_path, "logs", "aggregator_metrics.txt")
    model_accuracy = get_aggregated_accuracy(metric_file_path)
    log.info(f"Model accuracy post {request.config.num_rounds} rounds: {model_accuracy}")

    # Create new workspace with evaluation scope
    new_fed_obj = create_tr_dws_workspace(request, eval_scope=True)

    results_new = fed_helper.run_federation_for_dws(new_fed_obj, use_tls=request.config.use_tls)
    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        new_fed_obj,
        results_new,
        test_env=request.config.test_env,
        num_rounds=1,
    ), "Federation completion failed"

    new_metric_file_path = os.path.join(new_fed_obj.aggregator.workspace_path, "logs", "aggregator_metrics.txt")
    model_accuracy_eval = get_aggregated_accuracy(new_metric_file_path)
    log.info(f"Model accuracy during evaluation only on prev trained model is : {model_accuracy_eval})")

    # verify that the model accuracy is similar to the previous model accuracy max of 1% difference
    assert abs(model_accuracy - model_accuracy_eval) <= 0.01, "Model accuracy is not similar to the previous model accuracy"
