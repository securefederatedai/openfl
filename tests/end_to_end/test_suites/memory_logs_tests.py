# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import os

from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


@pytest.mark.log_memory_usage
def test_log_memory_usage(request, fx_federation):
    """
    Test for torch_cnn_mnist model.
    """
    # Skip test if fx_federation.log_memory_usage is False
    if not request.config.log_memory_usage:
        pytest.skip("Memory usage logging is disabled")

    # Setup PKI for trusted communication within the federation
    if request.config.use_tls:
        assert fed_helper.setup_pki(fx_federation), "Failed to setup PKI for trusted communication"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results, \
                                num_rounds=request.config.num_rounds), "Federation completion failed"
    # Verify the aggregator memory logs
    aggregator_log_file = os.path.join(fx_federation.workspace_path, "aggregator.log")

    memory_usage_dict = fed_helper.extract_memory_usage(aggregator_log_file)
    # check memory usage entries for each round
    assert len(memory_usage_dict) == request.config.num_rounds, \
                "Memory usage details are not available for all rounds"

    # check memory usage entries for each collaborator
    for collaborator in fx_federation.collaborators:
        collaborator_log_file = os.path.join(fx_federation.workspace_path, f"{collaborator.collaborator_name}.log")
        memory_usage_dict = fed_helper.extract_memory_usage(collaborator_log_file)
        assert len(memory_usage_dict) == request.config.num_rounds, \
                f"Memory usage details are not available for all rounds for collaborator {collaborator.collaborator_name}"
