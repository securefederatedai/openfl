# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.end_to_end.utils.logger import logger as log
from tests.end_to_end.utils import federation_helper as fed_helper


@pytest.mark.torch_cnn_mnist
def test_torch_cnn_mnist(fx_federation):
    """
    Test for torch_cnn_mnist model.
    """
    log.info(f"Test for torch_cnn_mnist with fx_federation: {fx_federation}")

    # Perform CSR operations like generating sign request, certifying request, etc.
    assert fed_helper.setup_pki(fx_federation), "Failed to perform CSR operations"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"


@pytest.mark.keras_cnn_mnist
def test_keras_cnn_mnist(fx_federation):
    log.info(f"Test for keras_cnn_mnist with fx_federation: {fx_federation}")

    # Perform CSR operations like generating sign request, certifying request, etc.
    assert fed_helper.setup_pki(fx_federation), "Failed to perform CSR operations"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"


@pytest.mark.torch_cnn_histology
def test_torch_cnn_histology(fx_federation):
    """
    Test for torch_cnn_histology model
    """
    log.info(f"Test for torch_cnn_histology with fx_federation: {fx_federation}")

    # Perform CSR operations like generating sign request, certifying request, etc.
    assert fed_helper.setup_pki(fx_federation), "Failed to perform CSR operations"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"
