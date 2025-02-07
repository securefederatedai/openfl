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

    if action == "stop":
        cmd_for_process_kill = constants.AGG_START_CMD if participant.name == "aggregator" else constants.COL_START_CMD.format(participant.name)
        pids = []
        # Find the process ID
        for line in os.popen(f"ps ax | grep '{cmd_for_process_kill}' | grep -v grep"):
            fields = line.split()
            pids.append(fields[0])

        if not pids:
            raise RuntimeError(f"No processes found for command '{cmd_for_process_kill}'")

        # Kill all processes using sudo
        for pid in pids:
            try:
                subprocess.run(['sudo', 'kill', '-9', pid], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to kill process '{pid}': {e}")

    else:
        try:
            participant.start(res_file=participant.res_file)
        except Exception as e:
            raise ex.ParticipantStartException(f"Error starting participant: {e}")

    return True
