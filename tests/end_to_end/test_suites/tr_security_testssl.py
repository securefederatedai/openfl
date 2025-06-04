# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import subprocess
import os
import json

from tests.end_to_end.utils.tr_common_fixtures import (
    fx_federation_tr,
)
from tests.end_to_end.utils import federation_helper as fed_helper
from tests.end_to_end.utils import defaults

log = logging.getLogger(__name__)


@pytest.mark.task_runner_basic
def test_federation_via_native(request, fx_federation_tr):
    """
    Test federation via native task runner.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    # Get aggregator address and port from plan.yaml
    plan_dir = defaults.AGG_PLAN_PATH.format(fx_federation_tr.local_bind_path)
    plan_file = os.path.join(plan_dir, "plan.yaml")
    aggreagtor_addr, aggregator_port = fed_helper.get_agg_addr_port(plan_file)

    # Run testssl.sh on the aggregator port
    output_path = os.path.join(fx_federation_tr.workspace_path, "testssl_output.json")
    run_testssl_sh(aggreagtor_addr, aggregator_port, output_path)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"

    best_agg_score = fed_helper.get_best_agg_score(fx_federation_tr.aggregator.tensor_db_file)
    log.info(f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}")
    # Verify the testssl.sh report
    verify_testssl_report(output_path)


def run_testssl_sh(aggregator_host, aggregator_port, output_path):
    """
    Run testssl.sh on the aggregator port.
    Args:
        aggregator_host (str): Aggregator host
        aggregator_port (int): Aggregator port
        output_path (str): Path to store the testssl.sh output
    """
    # Use testssl.sh to scan the aggregator port using subprocess and store the output in json file
    command = f"testssl --full --jsonfile {output_path} {aggregator_host}:{aggregator_port}"
    log.info(f"============== testssl.sh output for Aggregator - {aggregator_host}:{aggregator_port} ==============")
    subprocess.run(command, shell=True)


def verify_testssl_report(output_path):
    """
    Verify the testssl.sh report for security risks. If severity is HIGH, log the issue.
    Args:
        output_path (str): Path to testssl.sh output file
    """
    # Verify the testssl.sh report
    log.info("Verifying testssl.sh report")

    # Check if the testssl.sh output file exists
    assert os.path.exists(output_path), "testssl.sh output file not found"

    # Load the JSON output file
    with open(output_path, "r") as file:
        testssl_output = json.load(file)

    security_risk = False

    # Iterate through the testssl output items
    for item in testssl_output:
        # Check for high severity issues
        if item['severity'] == 'HIGH':
            # Skip certain checks that are not relevant due to internal CA usage
            if 'cipher_order' in item['id'] or 'cert_revocation' in item['id']:
                continue
            else:
                # Mark as security risk and log the issue
                security_risk = True
                log.error(f"Security risk found in testssl.sh report: {item}")

    # Assert that no security risks were found
    assert not security_risk, "testssl.sh report shows security risk"
