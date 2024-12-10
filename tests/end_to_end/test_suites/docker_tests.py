# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


@pytest.mark.docker
def test_federation_via_docker(request, fx_federation):
    """
    Test federation via docker.
    Args:
        request (Fixture): Pytest fixture
        fx_federation (Fixture): Pytest fixture
    """
    # Setup PKI for trusted communication within the federation
    if request.config.use_tls:
        assert fed_helper.setup_pki(fx_federation), "Failed to setup PKI for trusted communication"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results, request.config.num_rounds), "Federation completion failed"
