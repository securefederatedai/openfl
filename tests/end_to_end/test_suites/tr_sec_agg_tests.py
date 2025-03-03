# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
pass
pass
import pandas as pd
import json
import os

from tests.end_to_end.utils.tr_common_fixtures import (
    fx_federation_tr,
    fx_federation_tr_dws,
)
from tests.end_to_end.utils import db_helper as db_helper
from tests.end_to_end.utils import docker_helper as docker_helper
from tests.end_to_end.utils import federation_helper as fed_helper
from tests.end_to_end.utils import interruption_helper as int_helper

log = logging.getLogger(__name__)


# IMPORTANT - Please run the resiliency scenarios with higher no of rounds.

@pytest.fixture(scope="function")
def fx_configure_secc_agg(request):
    """
    Fixture to configure the Percentage Policy Straggler for the test.
    Args:
        request (Fixture): Pytest fixture
    """
    request.config.secc_agg = True


@pytest.mark.task_runner_basic
def test_federation_via_native_with_sec_agg(request, fx_configure_secc_agg, fx_federation_tr):
    """
    Test federation with aggregator restart via native task runner.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    assert fed_helper.run_federation(fx_federation_tr)

    db_file = fx_federation_tr.aggregator.tensor_db_file

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    )
    # Verify that metrics generated has masked value for collaborators.
    for collaborator in fx_federation_tr.collaborators:
        metric_file = os.path.join(fx_federation_tr.workspace_path, collaborator, "logs", f"{collaborator}_metrics.txt")
        assert verify_masked_metrics(metric_file)

    # Get the best aggregated score
    best_agg_score = fed_helper.get_best_agg_score(db_file)
    log.info(
        f"Model best aggregated score post {request.config.num_rounds} is {best_agg_score}"
    )

    log.info(
        f"Successfully tested federation experiment with multiple restart scenarios"
    )


def load_metrics_file(metric_file):
    """
    Load the metrics file into a pandas DataFrame.
    Args:
        metric_file (str): Path to the metric file
    Returns:
        pd.DataFrame: DataFrame containing the metrics data
    """
    with open(metric_file, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)


def validate_unmasked_difference(df, columns_to_check):
    """
    Validate that unmasked values are different from normal values.
    Args:
        df (pd.DataFrame): DataFrame containing the metrics data
        columns_to_check (list): List of columns to check for differences
    """
    def is_different(column):
        normal_col = column
        unmasked_col = f"{column}/unmasked"
        if normal_col in df.columns and unmasked_col in df.columns:
            differences = df[normal_col] != df[unmasked_col]
            assert differences.all(), f"Unmasked value is the same as normal value for {column}"
        else:
            raise ValueError(f"Columns {normal_col} or {unmasked_col} not found in DataFrame")

    for column in columns_to_check:
        is_different(column)


def verify_masked_metrics(metric_file):
    """
    Verify if the metrics values are masked.
    Args:
        metric_file (str): Path to the metric file
    """
    df = load_metrics_file(metric_file)
    columns_to_check = [
        "collaborator1/aggregated_model_validation/accuracy",
        "collaborator1/train/loss",
        "collaborator1/locally_tuned_model_validation/accuracy"
    ]
    validate_unmasked_difference(df, columns_to_check)
