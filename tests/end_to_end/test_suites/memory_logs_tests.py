# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import os
import json

from tests.end_to_end.utils.common_fixtures import fx_federation
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


@pytest.mark.log_memory_usage
def test_log_memory_usage(request, fx_federation):
    """
    This module contains end-to-end tests for logging memory usage in a federated learning setup.
    Test Suite:
        - test_log_memory_usage: Tests the memory usage logging functionality for the torch_cnn_mnist model.
    Functions:
    - test_log_memory_usage(request, fx_federation):
    Test the memory usage logging functionality in a federated learning setup.
    Parameters:
        - request: The pytest request object containing configuration options.
        - fx_federation: The fixture representing the federated learning setup.
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

    # Setup PKI for trusted communication within the federation
    if request.config.use_tls:
        assert fed_helper.setup_pki(
            fx_federation
        ), "Failed to setup PKI for trusted communication"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation, results, num_rounds=request.config.num_rounds
    ), "Federation completion failed"
    # Verify the aggregator memory logs
    aggregator_memory_usage_file = os.path.join(
        fx_federation.workspace_path,
        "aggregator",
        "workspace",
        "logs",
        "aggregator_memory_usage.json",
    )
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
    for collaborator in fx_federation.collaborators:
        collaborator_memory_usage_file = os.path.join(
            fx_federation.workspace_path,
            collaborator.name,
            "workspace",
            "logs",
            f"{collaborator.collaborator_name}_memory_usage.json",
        )

        assert os.path.exists(
            collaborator_memory_usage_file
        ), f"Memory usage file for collaborator {collaborator.collaborator_name} is not available"

        memory_usage_dict = json.load(open(collaborator_memory_usage_file))

        assert (
            len(memory_usage_dict) == request.config.num_rounds
        ), f"Memory usage details are not available for all rounds for collaborator {collaborator.collaborator_name}"

    log.info("Memory usage details are available for all participants")
