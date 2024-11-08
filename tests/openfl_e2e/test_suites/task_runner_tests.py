# Copyright 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.openfl_e2e.utils.logger import logger as log
from tests.openfl_e2e.utils import federation_helper as fed_helper


@pytest.mark.torch_cnn_mnist
def test_torch_cnn_mnist(fx_federation):
    """
    Test for torch_cnn_mnist model.
    """
    log.info(f"Test for torch_cnn_mnist with fx_federation: {fx_federation}")

    # Perform CSR operations like generating sign request, certifying request, etc.
    assert fed_helper.perform_csr_operations(fx_federation), "Failed to perform CSR operations"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"


@pytest.mark.keras_cnn_mnist
def test_keras_cnn_mnist(fx_federation):
    log.info(f"Test for keras_cnn_mnist with fx_federation: {fx_federation}")

    # Perform CSR operations like generating sign request, certifying request, etc.
    assert fed_helper.perform_csr_operations(fx_federation), "Failed to perform CSR operations"

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
    assert fed_helper.perform_csr_operations(fx_federation), "Failed to perform CSR operations"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"


@pytest.mark.tf_2dunet
def test_tf_2dunet(fx_federation):
    log.info(f"Test for tf_2dunet with fx_federation: {fx_federation}")

    # Perform CSR operations like generating sign request, certifying request, etc.
    assert fed_helper.perform_csr_operations(fx_federation), "Failed to perform CSR operations"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"


@pytest.mark.tf_cnn_histology
def test_tf_cnn_histology(fx_federation):
    log.info(f"Test for tf_cnn_histology with fx_federation: {fx_federation}")

    # Perform CSR operations like generating sign request, certifying request, etc.
    assert fed_helper.perform_csr_operations(fx_federation), "Failed to perform CSR operations"

    # Start the federation
    results = fed_helper.run_federation(fx_federation)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation, results), "Federation completion failed"
