# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import json
import re

import tests.end_to_end.utils.defaults as defaults
import tests.end_to_end.utils.interruption_helper as intr_helper
import tests.end_to_end.utils.ssh_helper as ssh

log = logging.getLogger(__name__)

def remove_stale_processes(aggregator=None, collaborators=[], director=None, envoys=[]):
    """
    Remove stale processes
    Args:
        aggregator (object): Aggregator object
        collaborators (list): List of collaborator objects
        director (object): Director object
        envoys (list): List of envoy objects
    """
    if aggregator:
        intr_helper.kill_processes(aggregator.name)

    for collaborator in collaborators:
        intr_helper.kill_processes(collaborator.name)

    if director:
        intr_helper.kill_processes("director")

    for envoy in envoys:
        intr_helper.kill_processes(envoy)

    log.info("Stale processes (if any) removed successfully")


def run_command(
    command,
    workspace_path,
    error_msg=None,
    container_id=None,
    run_in_background=False,
    bg_file=None,
    print_output=False,
    with_docker=False,
    return_error=False,
):
    """
    Run the command
    Args:
        command (str): Command to run
        workspace_path (str): Workspace path
        container_id (str): Container ID
        run_in_background (bool): Run the command in background
        bg_file (str): Background file (with path)
        print_output (bool): Print the output
        with_docker (bool): Flag specific to dockerized workspace scenario. Default is False.
        return_error (bool): Return error message
    Returns:
        tuple: Return code, output and error
    """
    return_code, output, error = 0, None, None
    error_msg = error_msg or "Failed to run the command"

    if with_docker and container_id:
        log.debug("Running command in docker container")
        if len(workspace_path):
            docker_command = f"docker exec -w {workspace_path} {container_id} sh -c "
        else:
            # This scenario is mainly for workspace creation where workspace path is not available
            docker_command = f"docker exec -i {container_id} sh -c "

        if run_in_background and bg_file:
            docker_command += f"'{command} > {bg_file}' &"
        else:
            docker_command += f"'{command}'"

        command = docker_command
    else:
        if not run_in_background:
            # When the command is run in background, we anyways pass the workspace path
            command = f"cd {workspace_path}; {command}"

    if print_output:
        log.info(f"Running command: {command}")

    if run_in_background and not with_docker:
        if bg_file:
            bg_file = open(bg_file, "a", buffering=1) # open file in append mode, so that restarting scenarios can be handled
        ssh.run_command_background(
            command,
            work_dir=workspace_path,
            redirect_to_file=bg_file,
            check_sleep=60,
        )
    else:
        return_code, output, error = ssh.run_command(command)
        if return_code != 0 and not return_error:
            log.error(f"{error_msg}: {error}")
            raise Exception(f"{error_msg}: {error}")

    if print_output:
        log.info(f"Output: {output}")
        log.info(f"Error: {error}")
    return return_code, output, error


# This functionality is common across multiple participants, thus moved to a common function
def verify_cmd_output(
    output, return_code, error, error_msg, success_msg, raise_exception=True
):
    """
    Verify the output of fx command run
    Assumption - it will have '✔️ OK' in the output if the command is successful
    Args:
        output (list): Output of the command using run_command()
        return_code (int): Return code of the command
        error (list): Error message
        error_msg (str): Error message
        success_msg (str): Success message
    """
    msg_received = [line for line in output if defaults.SUCCESS_MARKER in line]
    log.info(f"Message received: {msg_received}")
    if return_code == 0 and len(msg_received):
        log.info(success_msg)
    else:
        log.error(f"{error_msg}: {error}")
        if raise_exception:
            raise Exception(f"{error_msg}: {error}")


# TODO - remove if not needed in the near future
def extract_memory_usage(log_file):
    """
    Extracts memory usage data from a log file.
    This function reads the content of the specified log file, searches for memory usage data
    using a regular expression pattern, and returns the extracted data as a dictionary.
    Args:
        log_file (str): The path to the log file from which to extract memory usage data.
    Returns:
        dict: A dictionary containing the memory usage data.
    Raises:
        json.JSONDecodeError: If there is an error decoding the JSON data.
        Exception: If memory usage data is not found in the log file.
    """
    try:
        with open(log_file, "r") as file:
            content = file.read()

        pattern = r"Publish memory usage: (\[.*?\])"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            memory_usage_data = match.group(1)
            memory_usage_data = re.sub(r"\S+\.py:\d+", "", memory_usage_data)
            memory_usage_data = memory_usage_data.replace("\n", "").replace(" ", "")
            memory_usage_data = memory_usage_data.replace("'", '"')
            memory_usage_dict = json.loads(memory_usage_data)
            return memory_usage_dict
        else:
            log.error("Memory usage data not found in the log file")
            raise Exception("Memory usage data not found in the log file")
    except Exception as e:
        log.error(f"An error occurred while extracting memory usage: {e}")
        raise e


# TODO - remove if not needed in the near future
def write_memory_usage_to_file(memory_usage_dict, output_file):
    """
    Writes memory usage data to a file.
    This function writes the specified memory usage data to the specified output file.
    Args:
        memory_usage_dict (dict): A dictionary containing the memory usage data.
        output_file (str): The path to the output file to which to write the memory usage data.
    """
    try:
        with open(output_file, "w") as file:
            json.dump(memory_usage_dict, file, indent=4)
    except Exception as e:
        log.error(f"An error occurred while writing memory usage data to file: {e}")
        raise e
