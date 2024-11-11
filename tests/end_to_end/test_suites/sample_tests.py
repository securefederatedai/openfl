# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.end_to_end.utils.logger import logger as log
from tests.end_to_end.utils import federation_helper as fed_helper


# ** IMPORTANT **: This is just an example on how to add a test with below pre-requisites.
# Task Runner API Test function for federation run using sample_model
# 1. Create OpenFL workspace, if not present for the model and add relevant dataset and its path in plan/data.yaml
# 2. Append the model name to ModelName enum in tests/end_to_end/utils/constants.py
# 3. Add the model name to tests/end_to_end/pytest.ini marker, if not present
# 4. Use fx_federation fixture in the test function - it will provide the federation object.
# 5. Fixture will contain - model_owner, aggregator, collaborators, model_name, workspace_path, results_dir
# 6. Setup PKI for trusted communication within the federation
# 7. Start the federation using aggregator and given no of collaborators.
# 8. Verify the completion of the federation run.

@pytest.mark.sample_model
def test_sample_model(fx_federation):
    """
    Add a proper docstring here.
    """
    log.info(f"Running sample model test {fx_federation.model_name}")
    # Setup PKI for trusted communication within the federation
    assert fed_helper.setup_pki(fx_federation), "Failed to setup PKI"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"
