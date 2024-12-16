# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import os
import json

from tests.end_to_end.utils.common_fixtures import fx_federation_tr
import tests.end_to_end.utils.constants as constants
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)

# Note: This test file contains the test cases for logging memory usage in a federated learning setup.
# Fixture and marker mapping:
# fx_federation_tr - task_runner_basic and task_runner_docker
# fx_federation_tr_dws - task_runner_dockerized_ws

@pytest.mark.task_runner_basic
@pytest.mark.log_memory_usage
def test_log_memory_usage(request, fx_federation_tr):
    """
    This module contains end-to-end tests for logging memory usage in a federated learning setup.
    Test Suite:
        - test_log_memory_usage: Tests the memory usage logging functionality for the torch_cnn_mnist model.
    Functions:
    - test_log_memory_usage(request, fx_federation):
    Test the memory usage logging functionality in a federated learning setup.
    Parameters:
        - request: The pytest request object containing configuration options.
        - fx_federation_tr: The fixture representing the federated learning setup.
    Steps:
        1. Skip the test if memory usage logging is disabled.
        2. Setup PKI for trusted communication if TLS is enabled.
        3. Start the federation and verify its completion.
        4. Verify the existence of memory usage logs for the aggregator.
        5. Verify the memory usage details for each round.
        6. Verify the existence and details of memory usage logs for each collaborator.
        7. Log the availability of memory usage details for all participants.
    """
    # Skip test if fx_federation.log_memory_usage is False
    if not request.config.log_memory_usage:
        pytest.skip("Memory usage logging is disabled")

    # Start the federation
    results = fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr, results, num_rounds=request.config.num_rounds
    ), "Federation completion failed"

    # Verify the aggregator memory logs
    aggregator_memory_usage_file = constants.AGG_MEM_USAGE_JSON.format(fx_federation_tr)
    assert os.path.exists(
        aggregator_memory_usage_file
    ), "Aggregator memory usage file is not available"

    # Log the aggregator memory usage details
    memory_usage_dict = json.load(open(aggregator_memory_usage_file))

    # check memory usage entries for each round
    assert (
        len(memory_usage_dict) == request.config.num_rounds
    ), "Memory usage details are not available for all rounds"

    # check memory usage entries for each collaborator
    for collaborator in fx_federation_tr.collaborators:
        collaborator_memory_usage_file = constants.COL_MEM_USAGE_JSON.format(
            fx_federation_tr.workspace_path, collaborator.name
        )

        assert os.path.exists(
            collaborator_memory_usage_file
        ), f"Memory usage file for collaborator {collaborator.collaborator_name} is not available"

        memory_usage_dict = json.load(open(collaborator_memory_usage_file))

        assert (
            len(memory_usage_dict) == request.config.num_rounds
        ), f"Memory usage details are not available for all rounds for collaborator {collaborator.collaborator_name}"

    log.info("Memory usage details are available for all participants")
