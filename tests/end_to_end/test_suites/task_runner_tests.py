# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils.common_fixtures import fx_federation
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


@pytest.mark.torch_cnn_mnist
def test_torch_cnn_mnist(request, fx_federation):
    """
    Test for torch_cnn_mnist model.
    """
    log.info("Testing torch_cnn_mnist model")

    # Setup PKI for trusted communication within the federation
    if request.config.use_tls:
        assert fed_helper.setup_pki(fx_federation), "Failed to setup PKI for trusted communication"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results,
                                            num_rounds=request.config.num_rounds), "Federation completion failed"


@pytest.mark.keras_cnn_mnist
def test_keras_cnn_mnist(request, fx_federation):
    log.info("Testing keras_cnn_mnist model")

    # Setup PKI for trusted communication within the federation
    if request.config.use_tls:
        assert fed_helper.setup_pki(fx_federation), "Failed to setup PKI for trusted communication"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results,
                                            num_rounds=request.config.num_rounds), "Federation completion failed"


@pytest.mark.torch_cnn_histology
def test_torch_cnn_histology(request, fx_federation):
    """
    Test for torch_cnn_histology model
    """
    log.info("Testing torch_cnn_histology model")

    # Setup PKI for trusted communication within the federation
    if request.config.use_tls:
        assert fed_helper.setup_pki(fx_federation), "Failed to setup PKI for trusted communication"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results,
                                                num_rounds=request.config.num_rounds), "Federation completion failed"
