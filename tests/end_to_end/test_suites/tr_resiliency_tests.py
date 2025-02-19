# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import time
import math

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
def fx_configure_percentagepolicy(request):
    """
    Fixture to configure the Percentage Policy Straggler for the test.
    Args:
        request (Fixture): Pytest fixture
    """
    request.config.num_rounds = 30
    request.config.num_collaborators = 3
    request.config.model_name = "torch/mnist_straggler_check"
    request.config.straggler_policy ={
            "template": "openfl.component.aggregator.straggler_handling.PercentagePolicy",
            "settings": {
                "percent_collaborators_needed": 0.5,
                "minimum_reporting": 2
            }
        }


@pytest.fixture(scope="function")
def fx_configure_cutoffpolicy(request):
    """
    Fixture to configure the request CutoffTime policy straggler for the test.
    Args:
        request (Fixture): Pytest fixture
    """
    request.config.num_rounds = 30
    request.config.num_collaborators = 3
    request.config.model_name = "torch/mnist_straggler_check"
    request.config.straggler_policy ={
            "template": "openfl.component.aggregator.straggler_handling.CutoffTimePolicy",
            "settings": {
                "straggler_cutoff_time": 30,
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
def test_straggler_cutoff(request, fx_configure_cutoffpolicy, fx_federation_tr):
    """
    The cutoff policy in OpenFL stipulates that the aggregation process will happen
    with the 'minimum_reporting' number of collaborators if the remaining collaborators
    do not respond within the 'cutoff-time'. This means that aggregation could potentially
    happen with any number of collaborators, provided that the number is greater than the 'minimum_reporting' value.
    Args:
        request (Fixture): Pytest fixture
        fx_configure_cutoffpolicy (Fixture): Pytest fixture to configure the request cutoff for the test
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    db_file = fx_federation_tr.aggregator.tensor_db_file

    # Perform restart and validate rounds with stragglers
    minimum_reporting = request.config.straggler_policy["settings"]["minimum_reporting"]
    n_cols = request.config.num_collaborators - minimum_reporting

    # sleep for sometime before starting validation
    time.sleep(30)

    _perform_collaborator_restart_validate_rounds(
        fed_obj=fx_federation_tr,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
        min_reporting=minimum_reporting,
        n_cols=n_cols
    )
    log.info("Successfully tested minimum_reporting positive scenario")
    # sleep for sometime before starting validation
    time.sleep(30)

    _perform_collaborator_restart_validate_rounds(
        fed_obj=fx_federation_tr,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
        min_reporting=minimum_reporting,
        n_cols=n_cols+1
    )

    log.info("Successfully tested minimum_reporting negative scenario")

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
def test_straggler_percent_policy(request, fx_configure_percentagepolicy, fx_federation_tr):
    """
    The percentage policy in OpenFL ensures that the aggregation process
    always occurs with the 'minimum_reporting' number of collaborators and
    a satisfying percentage of collaborators. For instance, if there are a
    total of 5 collaborators, the 'minimum_reporting' value is 2, and the
    'percent_collaborators_needed' is 0.5, the aggregation process will always happen
    with at least 3 collaborators. It does not wait for the remaining 2
    collaborators to finish, as there's no specified cutoff time. In other
    words, the conditions for the percentage policy are met when both the
    'minimum_reporting' and 'percent_collaborators_needed' criteria are satisfied.
    Args:
        request (Fixture): Pytest fixture
        fx_configure_percentagepolicy (Fixture): Pytest fixture to
        configure the request percentage policy for the test
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    db_file = fx_federation_tr.aggregator.tensor_db_file

    # Retrieve the minimum reporting value from the configuration
    minimum_reporting = request.config.straggler_policy["settings"]["minimum_reporting"]

    # Calculate the required percentage of collaborators needed for reporting
    percentage_reporting = request.config.straggler_policy["settings"]["percent_collaborators_needed"]
    percentage_reporting = math.ceil(percentage_reporting * request.config.num_collaborators)

    # Ensure the minimum reporting value is at least the calculated percentage
    minimum_reporting = max(minimum_reporting, percentage_reporting)

    # Calculate the number of collaborators that can be restarted
    n_cols = request.config.num_collaborators - minimum_reporting

    # sleep for sometime before starting validation
    time.sleep(30)

    _perform_collaborator_restart_validate_rounds(
        fed_obj=fx_federation_tr,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
        min_reporting=minimum_reporting,
        n_cols=n_cols
    )
    log.info("Successfully tested minimum_reporting positive scenario")

    time.sleep(30)

    _perform_collaborator_restart_validate_rounds(
        fed_obj=fx_federation_tr,
        db_file=db_file,
        total_rounds=request.config.num_rounds,
        min_reporting=minimum_reporting,
        n_cols=n_cols+1
    )

    log.info("Successfully tested minimum_reporting negative scenario")

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


def _perform_collaborator_restart_validate_rounds(fed_obj, db_file, total_rounds, min_reporting, n_cols=1):
    """
    Perform collaborator restart and validate round increments.
        fed_obj (object): The federated learning object containing collaborators.
        db_file (str): The database file to track the current round.
        total_rounds (int): The total number of rounds to validate.
        min_reporting (int): The minimum number of collaborators to report.
        n_cols (int, optional): The number of collaborators to stop/start. Defaults to 1.
    """

    init_round = fed_helper.get_current_round(db_file)

    assert int_helper.restart_participants(fed_obj.collaborators[:n_cols], action="stop")

    log.info(f"{n_cols} Collaborators stopped successfully")

    round_increment = fed_helper.validate_round_increment(
        init_round,
        db_file,
        total_rounds,
        timeout=120,
    )

    # total number of collaborators - minimum reporting
    max_collaborators = len(fed_obj.collaborators)- min_reporting
    if n_cols <= max_collaborators:
        assert round_increment, f"Current round number is not increasing after {n_cols} collaborators stop."
        log.info(f"Current round number is increasing after {n_cols} collaborators stop as expected.")
    else:
        assert not round_increment, f"Current round number is increasing after {n_cols} collaborators stop. Expected to stop."
        log.info(f"Current round number is not increasing after {n_cols} collaborators stop as expected.")

    assert int_helper.restart_participants(fed_obj.collaborators[:n_cols], action="start")

    log.info(f"{n_cols} Collaborators restarted successfully")

    assert fed_helper.validate_round_increment(
        init_round,
        db_file,
        total_rounds,
        timeout=120,
    ), f"Expected current round to be ahead of {init_round} after collaborator restart"

    log.info("Current round number is increasing after every restart as expected.")
