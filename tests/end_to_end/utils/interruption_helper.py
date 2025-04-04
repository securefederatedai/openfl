# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import concurrent.futures
import time
import os
import subprocess   # nosec B404

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.docker_helper as docker_helper
import tests.end_to_end.utils.exceptions as ex

log = logging.getLogger(__name__)


def restart_participants(participants, action="restart") -> bool:
    """
    Restart the participant (collaborator or aggregator) in the federation.
    Args:
        participants: List of participant objects
        action: Action to perform (stop/start/restart). Default is restart.
    """
    if action not in ["stop", "start", "restart"]:
        raise ex.ParticipantRestartException(f"Invalid action {action}")

    executor = concurrent.futures.ThreadPoolExecutor()

    # ASSUMPTION - if container ID is present, it's docker environment else native

    if action in ["restart", "stop"]:
        # Stop the participants in parallel
        results = [
            executor.submit(
                stop_start_native_participant if participant.container_id is None else docker_helper.stop_start_docker_participant,
                participant,
                action="stop"
            )
            for participant in participants
        ]
        if not all([f.result() for f in results]):
            raise ex.ParticipantStopException(
                "Failed to stop one or more participants"
            )

    if action == "restart":
        # Wait for 10 seconds
        time.sleep(10)
        log.info("Waited for 10 seconds")

    if action in ["restart", "start"]:
        # Start the participants in parallel
        results = [
            executor.submit(
                stop_start_native_participant if participant.container_id is None else docker_helper.stop_start_docker_participant,
                participant,
                action="start"
            )
            for participant in participants
        ]
        if not all([f.result() for f in results]):
            raise ex.ParticipantStartException(
                "Failed to start one or more participants"
            )
    return True


def stop_start_native_participant(participant, action):
    """
    Function to stop/start given participant.
    Args:
        participant (object): Participant object
        action: Action to perform (stop/start)
    """
    if action not in ["stop", "start"]:
        raise ex.ParticipantStopException(f"Invalid action {action}")

    # Irrespective of the action, kill the processes to ensure clean state
    cmd_for_process = constants.AGG_START_CMD if participant.name == "aggregator" else constants.COL_START_CMD.format(participant.name)
    pids = []
    attempts = 5

    # Find the process ID and kill it
    try:
        kill_processes(cmd_for_process, fail_if_not_found=True)

    except subprocess.CalledProcessError:
        if action == "stop":
            raise RuntimeError(f"No processes found for command '{cmd_for_process}'")

    if action == "stop":
        log.info(f"Stopped {participant.name} successfully")
    else:
        try:
            participant.start()
            for i in range(1, attempts+1):
                pids = get_pids_if_command_running(cmd_for_process)
                if pids:
                    log.info(f"Participant '{participant.name}' started successfully with PIDs: {pids}")
                    break
                log.info(f"Waiting for participant '{participant.name}' to start... Attempt {i}/{attempts}")
                time.sleep(5)  # Wait for 1 second before retrying
            else:
                raise ex.ParticipantStartException(f"Participant {participant.name} failed to start")
        except Exception as e:
            raise ex.ParticipantStartException(f"Error starting participant {participant.name}: {e}")

    return True


def get_pids_if_command_running(command):
    """
    Get the process IDs of the given command if it is running.

    Args:
        command (str): The command to check.

    Returns:
        list: List of process IDs if the command is running, otherwise an empty list.
    """
    pids = []
    try:
        result = subprocess.run(f"ps -ef | grep '{command}' | grep -v grep", shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            # Extract the process IDs from the output
            pids = [line.split()[1] for line in result.stdout.strip().split('\n')]

    except subprocess.CalledProcessError as e:
        log.warning(f"Error checking for command '{command}': {e}")
    
    return pids


def kill_processes(command_to_kill, fail_if_not_found=False):
    """
    Kill all processes for the given command.
    
    Args:
        command_to_kill (str): The command to kill.
        fail_if_not_found (bool): Fail if given process is not found.
    
    Returns:
        bool: True if processes were killed, False otherwise.
    """
    try:
        pids = get_pids_if_command_running(command_to_kill)
        # Kill each process
        for pid in pids:
            subprocess.run(['sudo', 'kill', '-9', pid], check=fail_if_not_found)
        return True
    except subprocess.CalledProcessError:
        if fail_if_not_found:
            raise RuntimeError(f"Failed to kill process with PID {pid}")
        return False
