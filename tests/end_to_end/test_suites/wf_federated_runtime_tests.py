# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import os
import time
import concurrent.futures

import tests.end_to_end.utils.federation_helper as fh
import tests.end_to_end.utils.helper as helper
import tests.end_to_end.utils.wf_helper as wf_helper

log = logging.getLogger(__name__)


@pytest.mark.federated_runtime_301_watermarking
def test_federated_runtime_301_watermarking(request):
    """
    Test federated runtime without TLS.
    Args:
        request (Fixture): Pytest fixture
    """
    envoys = ["Bangalore", "Chandler"]
    workspace_path = os.path.join(
        os.getcwd(),
        "openfl-tutorials/experimental/workflow/FederatedRuntime/301_MNIST_Watermarking",
    )
    # Activate the experimental feature
    activate_experimental_feature(workspace_path)

    # Create result log files for the director and envoys
    result_path, participant_res_files = fh.create_federated_runtime_participant_res_files(
        request.config.results_dir, envoys, model_name="wf_watermarking"
    )

    # Start the director
    fh.start_director(workspace_path, participant_res_files["director"])

    # Start envoys Bangalore and Chandler and connect them to the director
    executor = concurrent.futures.ThreadPoolExecutor()
    results = [
        executor.submit(
            fh.start_envoy,
            envoy_name=envoy,
            workspace_path=workspace_path,
            res_file=participant_res_files[envoy.lower()],
        )
        for envoy in envoys
    ]
    assert all([f.result() for f in results]), "Failed to start one or more envoys"

    # Based on the pattern, the envoys take time to connect to the director
    # Hence, adding a sleep of 10 seconds anyways.
    time.sleep(10)
    nb_workspace_path = os.path.join(workspace_path, "workspace")
    notebook_path = nb_workspace_path + "/" + "MNIST_Watermarking.ipynb"

    assert fh.check_envoys_director_conn_federated_runtime(
        notebook_path=notebook_path, expected_envoys=envoys
    ), "Envoys are not connected to the director"

    # IMP - Notebook 301 Watermarking has hard coded notebook path set, hence changing the directory
    # This might not be true for all notebooks, thus keeping it as a separate step
    os.chdir(nb_workspace_path)

    assert wf_helper.run_notebook(
        notebook_path=notebook_path,
        output_notebook_path=result_path + "/" + "MNIST_Watermarking_output.ipynb"
    ), "Notebook run failed"

    # Change the directory back to the original directory
    os.chdir(os.getcwd())

    assert fh.verify_federated_runtime_experiment_completion(
        participant_res_files
    ), "Experiment failed"

    log.info("Experiment completed successfully")


@pytest.mark.federated_runtime_secure_aggregation
def test_federated_runtime_secure_aggregation(request):
    """
    Test federated runtime with TLS.
    Args:
        request (Fixture): Pytest fixture
    """
    envoys = ["Bengaluru", "Portland"]
    workspace_path = os.path.join(
        os.getcwd(),
        "openfl-tutorials/experimental/workflow/SecAgg",
    )
    # Activate the experimental feature
    activate_experimental_feature(workspace_path)

    # Create result log files for the director and envoys
    result_path, participant_res_files = fh.create_federated_runtime_participant_res_files(
        request.config.results_dir, envoys, model_name="wf_secagg"
    )
    # Start the director
    fh.start_director(workspace_path, participant_res_files["director"])

    # Start envoys Bengaluru and Portland and connect them to the director
    executor = concurrent.futures.ThreadPoolExecutor()
    results = [
        executor.submit(
            fh.start_envoy,
            envoy_name=envoy,
            workspace_path=workspace_path,
            res_file=participant_res_files[envoy.lower()],
        )
        for envoy in envoys
    ]
    assert all([f.result() for f in results]), "Failed to start one or more envoys"

    # Based on the pattern, the envoys take time to connect to the director
    # Hence, adding a sleep of 10 seconds anyways.
    time.sleep(10)
    nb_workspace_path = os.path.join(workspace_path, "workspace")
    notebook_path = nb_workspace_path + "/" + "MNIST_SecAgg.ipynb"

    assert fh.check_envoys_director_conn_federated_runtime(
        notebook_path=notebook_path, expected_envoys=envoys
    ), "Envoys are not connected to the director"

    # IMP - Notebook MNIST_SecAgg.ipynb has hard coded notebook path set, hence changing the directory
    # This might not be true for all notebooks, thus keeping it as a separate step
    os.chdir(nb_workspace_path)

    assert wf_helper.run_notebook(
        notebook_path=notebook_path,
        output_notebook_path=result_path + "/" + "MNIST_SecAgg_output.ipynb"
    ), "Notebook run failed"

    # Change the directory back to the original directory
    os.chdir(os.getcwd())

    assert fh.verify_federated_runtime_experiment_completion(
        participant_res_files
    ), "Experiment failed"

    log.info("Experiment completed successfully")


def activate_experimental_feature(workspace_path):
    """
    Activate the experimental feature.
    Args:
        workspace_path (str): Path to the workspace
    """
    # Activate the experimental feature
    cmd = f"fx experimental activate"
    error_msg = "Failed to activate the experimental feature"
    return_code, output, error = helper.run_command(
        cmd,
        workspace_path=workspace_path,
        error_msg=error_msg,
        return_error=True,
    )

    if error:
        # Check if the experimental feature is already activated
        if [err for err in error if "No such command 'activate'" in err]:
            log.info("Experimental feature already activated. Ignore the error.")
        else:
            log.error(f"{error_msg}: {error}")
            raise Exception(error)

    log.info(f"Activated the experimental feature.")
