# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils.tr_common_fixtures import fx_federation_tr
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


@pytest.mark.task_runner_with_s3
def test_federation_with_s3_bucket(request, fx_federation_tr):
    """
    Test federation with S3 bucket. Model name - torch/histology_s3
    Steps:
    1. Start the minio server, create buckets for every collaborator.
    2. Download data using torch/histology dataloader and upload data to the buckets.
    3. Create a datasources.json file for each collaborator which will contain the S3 bucket and/or local datasources.
    4. Calculate hash for each collaborator's data (it generates hash.txt file under the data directory).
    5. Start the federation (internally the hash is verified as well).
    6. Verify the completion of the federation run.
    7. Verify the best aggregated score.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
        time_for_each_round=300,
    ), "Federation completion failed"

    best_agg_score = fed_helper.get_best_agg_score(fx_federation_tr.aggregator.tensor_db_file)
    log.info(f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}")
