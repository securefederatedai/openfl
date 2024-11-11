# Copyright 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import concurrent.futures

from tests.end_to_end.utils.logger import logger as log


def perform_csr_operations(fed_obj):
    """
    Perform CSR operations like generating sign request, certifying request, etc.
    Args:
        fed_obj (object): Federation fixture object
    Returns:
        bool: True if successful, else False
    """
    success = False
    # Aggregator operations
    try:
        log.info(f"Performing operations for {fed_obj.aggregator.name}")
        fed_obj.aggregator.generate_sign_request()
        fed_obj.aggregator.certify_request()
    except Exception as e:
        log.error(f"Failed to perform aggregator operations: {e}")
        raise e

    # Collaborator operations
    for collaborator in fed_obj.collaborators:
        try:
            log.info(f"Performing operations for {collaborator.collaborator_name}")
            collaborator.create_collaborator()
            collaborator.generate_sign_request()
            # Below step will add collaborator entries in cols.yaml file.
            fed_obj.aggregator.sign_collaborator_csr(collaborator.collaborator_name)
            collaborator.import_certify_csr()
        except Exception as e:
            log.error(f"Failed to perform collaborator operations: {e}")
            raise e
    success = True

    log.info("CSR operations completed successfully for all participants")
    return success


def run_federation(fed_obj):
    """
    Start the federation
    Args:
        fed_obj (object): Federation fixture object
    Returns:
        list: List of response files for all the participants
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    # As the collaborators will wait for aggregator to start, we need to start them in parallel.
    futures = [
        executor.submit(
            participant.start
        )
        for participant in fed_obj.collaborators + [fed_obj.aggregator]
    ]

    # Result will contain response files for all the participants.
    results = [f.result() for f in futures]
    return results


def verify_federation_run_completion(fed_obj, results):
    """
    Verify the completion of the process for all the participants
    Args:
        fed_obj (object): Federation fixture object
        results (list): List of results
    Returns:
        list: List of response (True or False) for all the participants
    """
    log.info("Verifying the completion of the process for all the participants")
    # Start the collaborators and aggregator
    executor = concurrent.futures.ThreadPoolExecutor()
    # As the collaborators will wait for aggregator to start, we need to start them in parallel.
    futures = [
        executor.submit(
            _verify_completion_for_participant,
            participant,
            results[i]
        )
        for i, participant in enumerate(fed_obj.collaborators + [fed_obj.aggregator])
    ]

    # Result will contain a list of tuple of replica and operator objects.
    results = [f.result() for f in futures]
    log.info(f"Results: {results}")

    # If any of the participant failed, return False, else return True
    return all(results)


def _verify_completion_for_participant(participant, result_file):
    """
    Verify the completion of the process for the participant
    Args:
        participant (object): Participant object
        result_file (str): Result file
    Returns:
        bool: True if successful, else False
    """
    # Wait for the successful output message to appear in the log till timeout
    timeout = 100000 # in seconds
    log.info(f"Printing the last line of the log file for {participant.name} to track the progress")
    with open(result_file, 'r') as file:
        content = file.read()
    start_time = time.time()
    while (
        "OK" not in content and time.time() - start_time < timeout
    ):
        with open(result_file, 'r') as file:
            content = file.read()
        # Print last 2 lines of the log file on screen to track the progress
        log.info(f"{participant.name}: {content.splitlines()[-1:]}")
        if "OK" in content:
            break
        log.info(f"Process is yet to complete for {participant.name}")
        time.sleep(45)

    if "OK" not in content:
        log.error(f"Process failed/is incomplete for {participant.name} after timeout of {timeout} seconds")
        return False
    else:
        log.info(f"Process completed for {participant.name} in {time.time() - start_time} seconds")
        return True
