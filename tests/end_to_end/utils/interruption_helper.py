# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import concurrent.futures
import time
import psutil
import subprocess   # nosec B404

import tests.end_to_end.utils.defaults as defaults
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
    log.debug(f"Killing the processes (if running) for {participant.name} to avoid conflicts")
    participant.kill_process()

    if action == "stop":
        log.info(f"Stopped {participant.name} successfully")
    else:
        try:
            participant.start()
            log.info(f"Started {participant.name} successfully")
        except Exception as e:
            raise ex.ParticipantStartException(f"Error starting participant {participant.name}: {e}")

    return True


def get_pids_for_active_command(command):
    """
    Get the process IDs of the given command if it is running.

    Args:
        command (str): The command to check.

    Returns:
        list: List of process IDs if the command is running, otherwise an empty list.
    """
    pids = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if isinstance(cmdline, list):
                cmdline = ' '.join(cmdline)
                if command in cmdline:
                    pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
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
        pids = get_pids_for_active_command(command_to_kill)
        if len(pids):
            log.info(f"PIDs for command '{command_to_kill}': {pids}")
            # Kill each process
            for pid in pids:
                subprocess.run(['sudo', 'kill', '-9', str(pid)], check=fail_if_not_found)
                log.info(f"Killed process with PID {pid}")
        return True
    except subprocess.CalledProcessError:
        if fail_if_not_found:
            raise RuntimeError(f"Failed to kill process with PID {pid}")
        return False
