# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils.tr_common_fixtures import (
    fx_federation_tr,
    fx_federation_tr_dws,
)
from tests.end_to_end.utils import db_helper as db_helper
from tests.end_to_end.utils import docker_helper as docker_helper
from tests.end_to_end.utils import federation_helper as fed_helper
from tests.end_to_end.utils import interruption_helper as int_helper

log = logging.getLogger(__name__)


# IMPORTANT - Please run the resiliency scenarios with higher no of rounds.


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

    db_file = fx_federation_tr.aggregator.tensor_db_file
    _perform_restart_validate_rounds(fed_obj=fx_federation_tr, db_file=db_file, total_rounds=request.config.num_rounds)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    )

    best_agg_score = fed_helper.get_best_agg_score(db_file)
    log.info(
        f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}"
    )

    log.info(
        f"Successfully tested federation experiment with multiple restart scenarios"
    )


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

    db_file = fx_federation_tr_dws.aggregator.tensor_db_file
    _perform_restart_validate_rounds(fed_obj=fx_federation_tr_dws, db_file=db_file, total_rounds=request.config.num_rounds)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_dws,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    )

    best_agg_score = fed_helper.get_best_agg_score(db_file)
    log.info(
        f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}"
    )

    log.info(
        f"Successfully tested federation experiment with multiple restart scenarios"
    )


def _perform_restart_validate_rounds(fed_obj, db_file, total_rounds):
    """
    Internal function to perform restart and validate rounds.
    Args:
        fed_obj (Fixture): Pytest fixture for federation
        db_file (str): Path to the database file
        total_rounds (int): Total number of rounds
    """

    init_round = fed_helper.get_current_round(db_file)

    # Restart aggregator
    assert int_helper.restart_participants([fed_obj.aggregator])
    log.info("Aggregator restarted successfully")

    assert (
        round_post_agg_restart := fed_helper.validate_round_increment(
            init_round, db_file
        )
    ), f"Expected current round to be ahead of {init_round} after aggregator restart"

    # Restart collaborators
    assert int_helper.restart_participants(fed_obj.collaborators)
    log.info("Collaborators restarted successfully")

    assert (
        round_post_collab_restart := fed_helper.validate_round_increment(
            round_post_agg_restart, db_file
        )
    ), f"Expected current round to be ahead of {round_post_agg_restart} after collaborators restart"

    # Restart all participants
    assert int_helper.restart_participants(fed_obj.collaborators + [fed_obj.aggregator])
    log.info("All participants restarted successfully")

    assert fed_helper.validate_round_increment(
        round_post_collab_restart, db_file,
        total_rounds,
    ), f"Expected current round to be ahead of {round_post_collab_restart} after all participants restart"

    log.info("Current round number is increasing after every restart as expected.")
