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

@pytest.fixture(scope="function")
def fx_configure_request_percentagepolicy(request):
    """
    Fixture to configure the request cutoff for the test.
    Args:
        request (Fixture): Pytest fixture
    """
    request.config.num_rounds = 30
    request.config.num_collaborators = 3
    request.config.model_name = "torch/mnist_straggler_check"
    request.config.straggler_cutoff ={
            "template": "openfl.component.aggregator.straggler_handling.PercentagePolicy",
            "settings": {
                "percent_collaborators_needed": 0.5,
                "minimum_reporting": 2
            }
        }


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

    _perform_restart_validate_rounds(
        fed_obj=fx_federation_tr,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
    )

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


@pytest.mark.straggler_tests
def test_straggler_tests(request, fx_configure_request_percentagepolicy, fx_federation_tr):
    """
    Test federation with stragglers
    Args:
        request (Fixture): Pytest fixture
        fx_configure_request_percentagepolicy (Fixture): Pytest fixture to configure the request cutoff for the test
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    db_file = fx_federation_tr.aggregator.tensor_db_file

    # Perform restart and validate rounds with stragglers
    minimum_reporting = request.config.straggler_cutoff["settings"]["minimum_reporting"]
    n =  request.config.num_collaborators - minimum_reporting

    _perform_collaborator_restart_validate_rounds(
        fed_obj=fx_federation_tr,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
        min_reporting=minimum_reporting,
        n=n
    )

    _perform_collaborator_restart_validate_rounds(
        fed_obj=fx_federation_tr,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
        min_reporting=minimum_reporting,
        n=n+1
    )
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

    _perform_restart_validate_rounds(
        fed_obj=fx_federation_tr_dws,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
    )

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
            init_round,
            db_file,
            total_rounds,
        )
    ), f"Expected current round to be ahead of {init_round} after aggregator restart"

    # Restart collaborators
    assert int_helper.restart_participants(fed_obj.collaborators)
    log.info("Collaborators restarted successfully")

    assert (
        round_post_collab_restart := fed_helper.validate_round_increment(
            round_post_agg_restart,
            db_file,
            total_rounds,
        )
    ), f"Expected current round to be ahead of {round_post_agg_restart} after collaborators restart"

    # Restart all participants
    assert int_helper.restart_participants(fed_obj.collaborators + [fed_obj.aggregator])
    log.info("All participants restarted successfully")

    assert fed_helper.validate_round_increment(
        round_post_collab_restart,
        db_file,
        total_rounds,
    ), f"Expected current round to be ahead of {round_post_collab_restart} after all participants restart"

    log.info("Current round number is increasing after every restart as expected.")


def _perform_collaborator_restart_validate_rounds(fed_obj, db_file, total_rounds, min_reporting, n=1):
    """
    Perform collaborator restart and validate round increments.

        fed_obj (object): The federated learning object containing collaborators.
        db_file (str): The database file to track the current round.
        total_rounds (int): The total number of rounds to validate.
        n (int, optional): The number of collaborators to restart. Defaults to 1.

        int: The initial round number before the restart.
    """

    init_round = fed_helper.get_current_round(db_file)
    log.info(f"Initial round number is {init_round}")

    assert int_helper.restart_participants(fed_obj.collaborators[:n], action="stop")

    log.info(f"{n} Collaborators stopped successfully")

    round_increment =  fed_helper.validate_round_increment(
        init_round,
        db_file,
        total_rounds,
        timeout=120,
    ), f"Expected current round to be ahead of {init_round} after collaborator stop"

    # total number of collaborators - minimum reporting
    max_collaborators = len(fed_obj.collaborators)- min_reporting

    if n <= max_collaborators:
        assert round_increment, f"Current round number is not increasing after {n} collaborators stop."
        log.info(f"Current round number is increasing after {n} collaborators stop as expected.")
    else:
        assert not round_increment, f"Current round number is increasing after {n} collaborators stop. Expected to stop."
        log.info(f"Current round number is not increasing after {n} collaborators stop as expected.")

    assert int_helper.restart_participants(fed_obj.collaborators[:n], action="start")

    log.info(f"{n} Collaborators restarted successfully")

    assert fed_helper.validate_round_increment(
        init_round,
        db_file,
        total_rounds,
        timeout=120,
    ), f"Expected current round to be ahead of {init_round} after collaborator restart"

    log.info("Current round number is increasing after every restart as expected.")
    return init_round
