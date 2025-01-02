# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import os
import json

from tests.end_to_end.utils.common_fixtures import fx_federation_tr, fx_federation_tr_dws
import tests.end_to_end.utils.constants as constants
from tests.end_to_end.utils import federation_helper as fed_helper, ssh_helper as ssh
from tests.end_to_end.utils.generate_report import generate_memory_report

log = logging.getLogger(__name__)


@pytest.mark.log_memory_usage
def test_log_memory_usage_basic(request, fx_federation_tr):
    """
    Test the memory usage logging functionality in a federated learning setup.
    Args:
        - request: The pytest request object containing configuration options.
        - fx_federation_tr: The fixture representing the federated learning setup.
    """
    if not request.config.log_memory_usage:
        pytest.skip("Memory usage logging is disabled")

    _log_memory_usage(request, fx_federation_tr)


@pytest.mark.log_memory_usage
def test_log_memory_usage_dockerized_ws(request, fx_federation_tr_dws):
    """
    Test the memory usage logging functionality in a federated learning setup.
    Args:
        - request: The pytest request object containing configuration options.
        - fx_federation_tr_dws: The fixture representing the federated learning setup with dockerized workspace.
    """
    if not request.config.log_memory_usage:
        pytest.skip("Memory usage logging is disabled")

    _log_memory_usage(request, fx_federation_tr_dws)


def _log_memory_usage(request, fed_obj):
    """
    Test the memory usage logging functionality in a federated learning setup.
    Steps:
        1. Setup PKI for trusted communication if TLS is enabled.
        2. Start the federation and verify its completion.
        3. Verify the existence of memory usage logs for the aggregator.
        4. Verify the memory usage details for each round.
        5. Verify the existence and details of memory usage logs for each collaborator.
        6. Log the availability of memory usage details for all participants.
    """
    # Start the federation
    if request.config.test_env == "task_runner_basic":
        results = fed_helper.run_federation(fed_obj)
    else:
        results = fed_helper.run_federation_for_dws(
            fed_obj, use_tls=request.config.use_tls
        )

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fed_obj, results, test_env=request.config.test_env, num_rounds=request.config.num_rounds
    ), "Federation completion failed"

    # Verify the aggregator memory logs
    aggregator_memory_usage_file = constants.AGG_MEM_USAGE_JSON.format(fed_obj.workspace_path)

    if request.config.test_env == "task_runner_dockerized_ws":
        ssh.copy_file_from_docker(
            "aggregator", f"/workspace/logs/aggregator_memory_usage.json", aggregator_memory_usage_file
        )

    assert os.path.exists(
        aggregator_memory_usage_file
    ), "Aggregator memory usage file is not available"

    # Log the aggregator memory usage details
    memory_usage_dict = _convert_to_json(aggregator_memory_usage_file)
    aggregator_path = os.path.join(fed_obj.workspace_path, "aggregator")
    generate_memory_report(memory_usage_dict, aggregator_path)

    # check memory usage entries for each round
    assert (
        len(memory_usage_dict) == request.config.num_rounds
    ), "Memory usage details are not available for all rounds"

    # check memory usage entries for each collaborator
    for collaborator in fed_obj.collaborators:
        collaborator_memory_usage_file = constants.COL_MEM_USAGE_JSON.format(
            fed_obj.workspace_path, collaborator.name
        )
        if request.config.test_env == "task_runner_dockerized_ws":
            ssh.copy_file_from_docker(
                collaborator.name, f"/workspace/logs/{collaborator.name}_memory_usage.json", collaborator_memory_usage_file
            )
        assert os.path.exists(
            collaborator_memory_usage_file
        ), f"Memory usage file for collaborator {collaborator.collaborator_name} is not available"

        memory_usage_dict = _convert_to_json(collaborator_memory_usage_file)
        collaborator_path = os.path.join(fed_obj.workspace_path, collaborator.name)
        generate_memory_report(memory_usage_dict, collaborator_path)

        assert (
            len(memory_usage_dict) == request.config.num_rounds
        ), f"Memory usage details are not available for all rounds for collaborator {collaborator.collaborator_name}"

    log.info("Memory usage details are available for all participants")


def _convert_to_json(file):
    """
    Reads a file containing JSON objects, one per line, and converts them into a list of parsed JSON objects.

    Args:
        file (str): The path to the file containing JSON objects.

    Returns:
        list: A list of parsed JSON objects.
    """
    with open(file, 'r') as infile:
        json_objects = infile.readlines()

    # Parse each JSON object
    parsed_json_objects = [json.loads(obj) for obj in json_objects]
    return parsed_json_objects
