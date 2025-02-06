# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import time

from tests.end_to_end.utils.tr_common_fixtures import (
    fx_federation_tr,
    fx_federation_tr_dws,
)
from tests.end_to_end.utils import db_helper as db_helper
from tests.end_to_end.utils import docker_helper as docker_helper
from tests.end_to_end.utils import federation_helper as fed_helper
from tests.end_to_end.utils import interruption_helper as int_helper
from tests.end_to_end.utils.summary_helper import get_best_agg_score

log = logging.getLogger(__name__)


@pytest.mark.task_runner_basic
def test_federation_via_native_with_restarts(request, fx_federation_tr):
    """
    Test federation with aggregator restart via native task runner.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    # Wait for 60 seconds before checking the current round
    time.sleep(60)

    current_round = fed_helper.get_current_round(fx_federation_tr.aggregator.tensor_db_path)

    # Restart aggregator
    assert int_helper.restart_participants([fx_federation_tr.aggregator])

    time.sleep(20)
    round_post_agg_restart = fed_helper.get_current_round(fx_federation_tr.aggregator.tensor_db_path)
    assert round_post_agg_restart >= current_round, "Current round is not updated after aggregator restart"

    # Restart collaborators
    assert int_helper.restart_participants(fx_federation_tr.collaborators)

    time.sleep(20)
    round_post_collab_restart = fed_helper.get_current_round(fx_federation_tr.aggregator.tensor_db_path)
    assert round_post_collab_restart >= round_post_agg_restart, "Current round is not updated after collaborators restart"

    # Restart all participants
    assert int_helper.restart_participants(fx_federation_tr.collaborators+[fx_federation_tr.aggregator])

    time.sleep(20)
    round_post_all_restart = fed_helper.get_current_round(fx_federation_tr.aggregator.tensor_db_path)
    assert round_post_all_restart >= round_post_collab_restart, "Current round is not updated after all participants restart"

    log.info("All restarts are successful with current round updated")

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    )

    model_accuracy = get_best_agg_score(fx_federation_tr.aggregator.tensor_db_path)
    log.info(f"Model accuracy post {request.config.num_rounds} rounds: {model_accuracy}")

    log.info(f"Successfully tested federation experiment with aggregator restart")


@pytest.mark.task_runner_dockerized_ws
def test_federation_via_dws_with_restarts(request, fx_federation_tr_dws):
    """
    Test federation via dockerized workspace.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr_dws (Fixture): Pytest fixture for dockerized workspace
    """
    # Start the federation
    fed_helper.run_federation_for_dws(fx_federation_tr_dws, request.config.use_tls)

    # Wait for 60 seconds before checking the current round
    time.sleep(60)

    current_round = fed_helper.get_current_round(fx_federation_tr_dws.aggregator.tensor_db_path)

    # Restart aggregator
    assert int_helper.restart_participants([fx_federation_tr_dws.aggregator])

    time.sleep(20)
    round_post_agg_restart = fed_helper.get_current_round(fx_federation_tr_dws.aggregator.tensor_db_path)
    assert round_post_agg_restart >= current_round, "Current round is not updated after aggregator restart"

    # Restart collaborators
    assert int_helper.restart_participants(fx_federation_tr_dws.collaborators)

    time.sleep(20)
    round_post_collab_restart = fed_helper.get_current_round(fx_federation_tr_dws.aggregator.tensor_db_path)
    assert round_post_collab_restart >= round_post_agg_restart, "Current round is not updated after collaborators restart"

    # Restart all participants
    assert int_helper.restart_participants(fx_federation_tr_dws.collaborators+[fx_federation_tr_dws.aggregator])

    time.sleep(20)
    round_post_all_restart = fed_helper.get_current_round(fx_federation_tr_dws.aggregator.tensor_db_path)
    assert round_post_all_restart >= round_post_collab_restart, "Current round is not updated after all participants restart"

    log.info("All restarts are successful with current round updated")

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_dws,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    )

    model_accuracy = get_best_agg_score(fx_federation_tr_dws.aggregator.tensor_db_path)
    log.info(f"Model accuracy post {request.config.num_rounds} rounds: {model_accuracy}")

    log.info(f"Successfully tested federation experiment with aggregator restart")

