# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import concurrent.futures

import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.utils.common_fixtures import fx_federation_tr_dws
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


# NOTE: This test file contains the test cases for the task runner federation using dockerized workspace approach.

@pytest.mark.task_runner_dockerized_ws
def test_federation_via_dockerized_workspace(request, fx_federation_tr_dws):
    """
    Test federation via dockerized workspace.
    Args:
        request (Fixture): Pytest fixture
        fx_federation (Fixture): Pytest fixture
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    try:
        results = [
            executor.submit(
                ssh.run_command,
                cmd="tar -xf /certs.tar",
                work_dir=participant.workspace_path,
            )
            for participant in [fx_federation_tr_dws.aggregator] + fx_federation_tr_dws.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to extract certificates for one or more participants")
    except Exception as e:
        raise e

    try:
        results = [
            executor.submit(
                collaborator.import_pki,
                zip_name=f"agg_to_col_{collaborator.name}_signed_cert.zip"
            )
            for collaborator in fx_federation_tr_dws.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to import and certify the CSR for one or more collaborators")
    except Exception as e:
        raise e

    # Start the federation
    results = fed_helper.run_federation(fx_federation_tr_dws)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(fx_federation_tr_dws, results, request.config.num_rounds), "Federation completion failed"
