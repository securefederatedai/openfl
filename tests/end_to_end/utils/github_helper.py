# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import logging
import time

log = logging.getLogger(__name__)


def fetch_args():
    """
    Function to fetch the commandline arguments.
    Returns:
        Parsed arguments
    """
    # Initialize the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--func_name", required=True, default="", type=str, help="Name of function to be called"
    )
    # All the arguments except for func_name are optional and have default values.
    parser.add_argument(
        "--notebook_path", required=False, default="", type=str, help="Path to the notebook including name"
    )
    parser.add_argument(
        "--expected_envoys", required=False, default="", type=str, help="List of expected envoy names. For e.g. ['Bangalore', 'Chandler']"
    )
    parser.add_argument(
        "--director_node_fqdn", required=False, default="localhost", type=str, help="Director node FQDN"
    )
    parser.add_argument(
        "--director_port", required=False, default=50050, type=int, help="Director port number"
    )
    args = parser.parse_args()
    return args


def verify_envoys_director_conn_federated_runtime(notebook_path, expected_envoys, director_node_fqdn="localhost", director_port=50050):
    """
    Function to get the envoys specifically for the Federated Runtime notebooks
    Args:
        notebook_path (str): Path to the notebook including name
        expected_envoys (list): List of expected envoy names
        director_node_fqdn (str): Director node FQDN (default is localhost)
        director_port (int): Director port number (default is 50050)
    """
    from openfl.experimental.workflow.runtime import FederatedRuntime
    # Based on the pattern, the envoys take time to connect to the director
    # Hence, adding a sleep of 10 seconds anyways.
    time.sleep(10)

    # Number of retries and delay between retries in seconds
    MAX_RETRIES = RETRY_DELAY = 5

    log.debug(f"Notebook path: {notebook_path}")
    expected_envoys = ast.literal_eval(expected_envoys)
    log.debug(f"Expected envoys: {expected_envoys}")

    director_info = {
        'director_node_fqdn': director_node_fqdn,
        'director_port': director_port,
    }

    federated_runtime = FederatedRuntime(
        collaborators=expected_envoys,
        director=director_info,
        notebook_path=notebook_path
    )
    # Retry logic
    for attempt in range(MAX_RETRIES):
        actual_envoys = federated_runtime.get_envoys()
        if all(sorted(expected_envoys) == sorted(actual_envoys) for expected_envoys, actual_envoys in [(expected_envoys, actual_envoys)]):
            log.info("All the envoys are connected to the director")
            return True
        else:
            log.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: Not all envoys are connected. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

    # If the loop completes without returning, raise an exception
    raise Exception("Not all envoys are connected to the director after multiple attempts")


if __name__ == "__main__":
    # Fetch input arguments
    args = fetch_args()
    func_name = args.func_name

    if func_name == "verify_envoys_director_conn_federated_runtime":
        verify_envoys_director_conn_federated_runtime(args.notebook_path, args.expected_envoys, args.director_node_fqdn, args.director_port)
