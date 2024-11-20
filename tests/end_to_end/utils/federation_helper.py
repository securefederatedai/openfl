# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import concurrent.futures
import logging

from tests.end_to_end.utils.constants import SUCCESS_MARKER

log = logging.getLogger(__name__)


def setup_pki(fed_obj):
    """
    Setup PKI for trusted communication within the federation

    Args:
        fed_obj (object): Federation fixture object
    Returns:
        bool: True if successful, else False
    """
    success = False
    # Aggregator and model owner operations
    try:
        log.info(f"Performing operations for {fed_obj.aggregator.name}")
        fed_obj.aggregator.generate_sign_request()
        fed_obj.model_owner.certify_aggregator(fed_obj.aggregator.agg_domain_name)
    except Exception as e:
        log.error(f"Failed to perform PKI setup for aggregator: {e}")
        raise e

    # Collaborator and model owner operations
    for collaborator in fed_obj.collaborators:
        try:
            log.info(f"Performing operations for {collaborator.collaborator_name}")
            collaborator.generate_sign_request()
            # Below step will add collaborator entries in cols.yaml file.
            fed_obj.model_owner.certify_collaborator(collaborator.collaborator_name)
            collaborator.import_pki()
        except Exception as e:
            log.error(f"Failed to perform PKI setup for {collaborator.collaborator_name}: {e}")
            raise e
    success = True

    log.info("PKI setup successfully for all participants")
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
            fed_obj.num_rounds,
            results[i]
        )
        for i, participant in enumerate(fed_obj.collaborators + [fed_obj.aggregator])
    ]

    # Result will contain a list of boolean values for all the participants.
    # True - successful completion, False - failed/incomplete
    results = [f.result() for f in futures]
    log.info(f"Results from all the participants: {results}")

    # If any of the participant failed, return False, else return True
    return all(results)


def _verify_completion_for_participant(participant, num_rounds, result_file, time_for_each_round=100):
    """
    Verify the completion of the process for the participant
    Args:
        participant (object): Participant object
        result_file (str): Result file
    Returns:
        bool: True if successful, else False
    """
    # Wait for the successful output message to appear in the log till timeout
    timeout = 300 + ( time_for_each_round * num_rounds ) # in seconds
    log.info(f"Printing the last line of the log file for {participant.name} to track the progress")
    with open(result_file, 'r') as file:
        content = file.read()
    start_time = time.time()
    while (
        SUCCESS_MARKER not in content and time.time() - start_time < timeout
    ):
        with open(result_file, 'r') as file:
            content = file.read()
        # Print last 2 lines of the log file on screen to track the progress
        log.info(f"{participant.name}: {content.splitlines()[-1:]}")
        if SUCCESS_MARKER in content:
            break
        log.info(f"Process is yet to complete for {participant.name}")
        time.sleep(45)

    if SUCCESS_MARKER not in content:
        log.error(f"Process failed/is incomplete for {participant.name} after timeout of {timeout} seconds")
        return False
    else:
        log.info(f"Process completed for {participant.name} in {time.time() - start_time} seconds")
        return True
