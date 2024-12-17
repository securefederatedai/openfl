# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import concurrent.futures
import logging
import os
import json
import re

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.ssh_helper as sh
from tests.end_to_end.models import collaborator as col_model

log = logging.getLogger(__name__)


def setup_pki(fed_obj):
    """
    Setup PKI for trusted communication within the federation

    Args:
        fed_obj (object): Federation fixture object
    Returns:
        bool: True if successful, else False
    """
    # PKI setup for aggregator is done at fixture level
    local_agg_ws_path = os.path.join(fed_obj.local_bind_path, "aggregator", "workspace")

    executor = concurrent.futures.ThreadPoolExecutor()

    # Performing all the operations step by step
    # This is to avoid problems during parallel execution
    # in case one or more collaborator operations delay for some reason
    # Generate sign request for all the collaborators
    try:
        results = [
            executor.submit(
                collaborator.generate_sign_request,
            )
            for collaborator in fed_obj.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to generate sign request for one or more collaborators")

    except Exception as e:
        raise e

    # Copy the generated sign request zip from all the collaborators to aggregator
    try:
        results = [
            executor.submit(
                copy_file_between_participants,
                local_src_path=os.path.join(fed_obj.local_bind_path, collaborator.name, "workspace"),
                local_dest_path=local_agg_ws_path,
                file_name=f"col_{collaborator.name}_to_agg_cert_request.zip"
            )
            for collaborator in fed_obj.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to copy sign request zip from one or more collaborators to aggregator")
    except Exception as e:
        raise e

    # Certify the collaborator sign requests
    # DO NOT run this in parallel as it causes command to fail with FileNotFoundError for a different collaborator
    for collaborator in fed_obj.collaborators:
        try:
            fed_obj.model_owner.certify_collaborator(
                collaborator_name=collaborator.name,
                zip_name=f"col_{collaborator.name}_to_agg_cert_request.zip"
            )
        except Exception as e:
            log.error(f"Failed to certify sign request for {collaborator.name}: {e}")
            raise e

    # Copy the signed certificates from aggregator to all the collaborators
    try:
        results = [
            executor.submit(
                copy_file_between_participants,
                local_src_path=local_agg_ws_path,
                local_dest_path=os.path.join(fed_obj.local_bind_path, collaborator.name, "workspace"),
                file_name=f"agg_to_col_{collaborator.name}_signed_cert.zip"
            )
            for collaborator in fed_obj.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to copy signed certificates from aggregator to one or more collaborators")
    except Exception as e:
        raise e

    # Import and certify the CSR for all the collaborators
    try:
        results = [
            executor.submit(
                collaborator.import_pki,
                zip_name=f"agg_to_col_{collaborator.name}_signed_cert.zip"
            )
            for collaborator in fed_obj.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to import and certify the CSR for one or more collaborators")
    except Exception as e:
        raise e

    # Copy the cols.yaml file from aggregator to all the collaborators
    # File cols.yaml is updated after PKI setup
    try:
        results = [
            executor.submit(
                copy_file_between_participants,
                local_src_path=os.path.join(local_agg_ws_path, "plan"),
                local_dest_path=os.path.join(fed_obj.local_bind_path, collaborator.name, "workspace", "plan"),
                file_name="cols.yaml",
                run_with_sudo=True,
            )
            for collaborator in fed_obj.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to copy cols.yaml file from aggregator to one or more collaborators")
    except Exception as e:
        raise e

    return True


def copy_file_between_participants(local_src_path, local_dest_path, file_name, run_with_sudo=False):
    """
    Copy file between participants
    Args:
        local_src_path (str): Source path on local machine
        local_dest_path (str): Destination path on local machine
        file_name (str): File name only (without path)
        run_with_sudo (bool): Run the command with sudo
    """
    cmd = "sudo cp" if run_with_sudo else "cp"
    cmd += f" {local_src_path}/{file_name} {local_dest_path}"
    return_code, output, error = sh.run_command(cmd)
    if return_code != 0:
        log.error(f"Failed to copy file: {error}")
        raise Exception(f"Failed to copy file: {error}")
    log.info(f"File {file_name} copied successfully from {local_src_path} to {local_dest_path}")
    return True


def run_federation(fed_obj):
    """
    Start the federation
    Args:
        fed_obj (object): Federation fixture object
    Returns:
        list: List of response files for all the participants
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    # Install dependencies on collaborators
    # This is a time taking process, thus doing at this stage after all verification is done
    log.info("Installing dependencies on collaborators. This might take some time...")
    futures = [
        executor.submit(
            participant.install_dependencies
        )
        for participant in fed_obj.collaborators
    ]
    results = [f.result() for f in futures]
    log.info(f"Results from all the collaborators for installation of dependencies: {results}")

    if not all(results):
        raise Exception("Failed to install dependencies on one or more collaborators")

    # As the collaborators will wait for aggregator to start, we need to start them in parallel.
    futures = [
        executor.submit(
            participant.start,
            os.path.join(fed_obj.workspace_path, participant.name, "workspace", f"{participant.name}.log")
        )
        for participant in fed_obj.collaborators + [fed_obj.aggregator]
    ]

    # Result will contain response files for all the participants.
    results = [f.result() for f in futures]
    if not all(results):
        raise Exception("Failed to start one or more participants")
    return results


def verify_federation_run_completion(fed_obj, results, num_rounds):
    """
    Verify the completion of the process for all the participants
    Args:
        fed_obj (object): Federation fixture object
        results (list): List of results
        num_rounds (int): Number of rounds
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
            num_rounds,
            results[i],
            local_bind_path=fed_obj.local_bind_path
        )
        for i, participant in enumerate(fed_obj.collaborators + [fed_obj.aggregator])
    ]

    # Result will contain a list of boolean values for all the participants.
    # True - successful completion, False - failed/incomplete
    results = [f.result() for f in futures]
    log.info(f"Results from all the participants: {results}")

    # If any of the participant failed, return False, else return True
    return all(results)


def _verify_completion_for_participant(participant, num_rounds, result_file, time_for_each_round=100, local_bind_path=None):
    """
    Verify the completion of the process for the participant
    Args:
        participant (object): Participant object
        num_rounds (int): Number of rounds
        result_file (str): Result file
        time_for_each_round (int): Time for each round
        local_bind_path (str, Optional): Local bind path. Applicable in case of docker environment
    Returns:
        bool: True if successful, else False
    """
    time.sleep(20) # Wait for some time before checking the log file
    # Set timeout based on the number of rounds and time for each round
    timeout = 600 + ( time_for_each_round * num_rounds ) # in seconds
    log.info(f"Printing the last line of the log file for {participant.name} to track the progress")

    # In case of docker environment, get the logs from local path which is mounted to the container
    if os.getenv("TEST_ENV") == "docker":
        result_file_name = os.path.basename(result_file)
        result_file = os.path.join(local_bind_path, participant.name, "workspace", result_file_name)

    log.info(f"Result file is: {result_file}")

    # Do not open file here as it will be opened in the loop below
    # Also it takes time for the federation run to start and write the logs
    content = [""]

    start_time = time.time()
    while (
        constants.SUCCESS_MARKER not in content and time.time() - start_time < timeout
    ):
        with open(result_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        content = list(filter(str.rstrip, lines))[-1:]

        # Print last line of the log file on screen to track the progress
        log.info(f"{participant.name}: {content}")
        if constants.SUCCESS_MARKER in content:
            break
        log.info(f"Process is yet to complete for {participant.name}")
        time.sleep(45)

    if constants.SUCCESS_MARKER not in content:
        log.error(f"Process failed/is incomplete for {participant.name} after timeout of {timeout} seconds")
        return False
    else:
        log.info(f"Process completed for {participant.name} in {time.time() - start_time} seconds")
        return True


def federation_env_setup_and_validate(request):
    """
    Setup the federation environment and validate the configurations
    Args:
        request (object): Request object
    Returns:
        tuple: Test environment, model name, workspace path, aggregator domain name
    """
    # Determine the test type based on the markers
    markers = [m.name for m in request.node.iter_markers()]
    os.environ["TEST_ENV"] = test_env = "docker" if "docker" in markers else "task_runner"

    # Validate the model name and create the workspace name
    if not request.config.model_name.upper() in constants.ModelName._member_names_:
        raise ValueError(f"Invalid model name: {request.config.model_name}")

    # Set the workspace path
    home_dir = os.getenv("HOME")

    if test_env == "docker":
        # First check if openfl image is available
        dh.check_docker_image()
        # Cleanup docker containers
        dh.cleanup_docker_containers()
        dh.remove_docker_network()

        # Create docker network openfl
        dh.create_docker_network()

        local_bind_path = os.path.join(home_dir, request.config.results_dir, request.config.model_name)
        # Absolute path is required for docker
        workspace_path = os.path.join("/", request.config.results_dir, request.config.model_name)
        agg_domain_name = "aggregator"
    else:
        local_bind_path = workspace_path = os.path.join(home_dir, request.config.results_dir, request.config.model_name)
        agg_domain_name = "localhost"

    log.info(
        f"Running federation setup using {test_env} API on single machine with below configurations:\n"
        f"\tNumber of collaborators: {request.config.num_collaborators}\n"
        f"\tNumber of rounds: {request.config.num_rounds}\n"
        f"\tModel name: {request.config.model_name}\n"
        f"\tClient authentication: {request.config.require_client_auth}\n"
        f"\tTLS: {request.config.use_tls}\n"
        f"\tMemory Logs: {request.config.log_memory_usage}\n"
        f"\tResults directory: {request.config.results_dir}\n"
        f"\tWorkspace path: {workspace_path}"
    )
    return test_env, request.config.model_name, workspace_path, local_bind_path, agg_domain_name


def add_local_workspace_permission(local_bind_path):
    """
    Add permission to workspace. This is aggregator/model owner specific operation.
    Args:
        workspace_path (str): Workspace path
        agg_container_id (str): Container ID
    """
    try:
        agg_workspace_path = os.path.join(local_bind_path, "aggregator", "workspace")
        return_code, output, error = run_command(
            f"sudo chmod -R 777 {agg_workspace_path}",
            workspace_path=local_bind_path,
        )
        if return_code != 0:
            raise Exception(f"Failed to add local permission to workspace: {error}")

        log.debug(f"Recursive permission added to workspace on local machine: {agg_workspace_path}")
    except Exception as e:
        log.error(f"Failed to add local permission to workspace: {e}")
        raise e


def create_persistent_store(participant_name, local_bind_path):
    """
    Create persistent store for the participant on local machine (even for docker)
    Args:
        participant_name (str): Participant name
        local_bind_path (str): Local bind path
    """
    try:
        # Create persistent store
        error_msg = f"Failed to create persistent store for {participant_name}"
        cmd_persistent_store = (
            f"export WORKING_DIRECTORY={local_bind_path}; " \
            f"mkdir -p $WORKING_DIRECTORY/{participant_name}/workspace; " \
            "sudo chmod -R 755 $WORKING_DIRECTORY"
        )
        log.debug(f"Creating persistent store")
        return_code, output, error = run_command(
            cmd_persistent_store,
            workspace_path=os.getenv("HOME"),
        )
        if error:
            raise ex.PersistentStoreCreationException(f"{error_msg}: {error}")

        log.info(f"Persistent store created for {participant_name}")

    except Exception as e:
        raise ex.PersistentStoreCreationException(f"{error_msg}: {e}")


def run_command(command, workspace_path, error_msg=None, container_id=None, run_in_background=False, bg_file=None, print_output=False):
    """
    Run the command
    Args:
        command (str): Command to run
        workspace_path (str): Workspace path
        container_id (str): Container ID
        run_in_background (bool): Run the command in background
        bg_file (str): Background file (with path)
        print_output (bool): Print the output
    Returns:
        tuple: Return code, output and error
    """
    return_code, output, error = 0, None, None
    error_msg = error_msg or "Failed to run the command"
    is_docker = True if os.getenv("TEST_ENV") == "docker" else False
    if is_docker and container_id:
        log.debug("Running command in docker container")
        if len(workspace_path):
            docker_command = f"docker exec -w {workspace_path} {container_id} sh -c "
        else:
            # This scenario is mainly for workspace creation where workspace path is not available
            docker_command = f"docker exec -i {container_id} sh -c "

        if run_in_background and bg_file:
            docker_command += f"'{command} > {bg_file} &'"
        else:
            docker_command += f"'{command}'"

        command = docker_command
    else:
        if not run_in_background:
            # When the command is run in background, we anyways pass the workspace path
            command = f"cd {workspace_path}; {command}"

    if print_output:
        log.info(f"Running command: {command}")

    log.debug("Running command on local machine")
    if run_in_background and not is_docker:
        bg_file = open(bg_file, "w", buffering=1)
        sh.run_command_background(
            command,
            work_dir=workspace_path,
            redirect_to_file=bg_file,
            check_sleep=60,
        )
    else:
        return_code, output, error = sh.run_command(command)
        if return_code != 0:
            log.error(f"{error_msg}: {error}")
            raise Exception(f"{error_msg}: {error}")

    if print_output:
        log.info(f"Output: {output}")
        log.info(f"Error: {error}")
    return return_code, output, error


# This functionality is common across multiple participants, thus moved to a common function
def verify_cmd_output(output, return_code, error, error_msg, success_msg, raise_exception=True):
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
    msg_received = [line for line in output if constants.SUCCESS_MARKER in line]
    log.info(f"Message received: {msg_received}")
    if return_code == 0 and len(msg_received):
        log.info(success_msg)
    else:
        log.error(f"{error_msg}: {error}")
        if raise_exception:
            raise Exception(f"{error_msg}: {error}")


def setup_collaborator(count, workspace_path, local_bind_path):
    """
    Setup the collaborator
    Args:
        count (int): Count of collaborator
        workspace_path (str): Workspace path
        local_bind_path (str): Local bind path
    Returns:
        object: Collaborator object
    """
    local_agg_ws_path = os.path.join(local_bind_path, "aggregator", "workspace")
    collaborator = None

    try:
        collaborator = col_model.Collaborator(
            collaborator_name=f"collaborator{count+1}",
            data_directory_path=count + 1,
            workspace_path=f"{workspace_path}/collaborator{count+1}/workspace",
        )
        create_persistent_store(collaborator.name, local_bind_path)

    except Exception as e:
        raise ex.PersistentStoreCreationException(f"Failed to create persistent store for {collaborator.name}: {e}")

    try:
        if os.getenv("TEST_ENV") == "docker":
            container = dh.start_docker_container(
                container_name=collaborator.name,
                workspace_path=workspace_path,
                local_bind_path=local_bind_path,
            )
            collaborator.container_id = container.id
    except Exception as e:
        raise ex.DockerException(f"Failed to start {collaborator.name} docker environment: {e}")

    try:
        local_col_ws_path = os.path.join(local_bind_path, collaborator.name, "workspace")
        copy_file_between_participants(local_agg_ws_path, local_col_ws_path, "workspace.zip")
        collaborator.import_workspace()
    except Exception as e:
        raise ex.WorkspaceImportException(f"Failed to import workspace for {collaborator.name}: {e}")

    try:
        collaborator.create_collaborator()
    except Exception as e:
        raise ex.CollaboratorCreationException(f"Failed to create collaborator: {e}")

    return collaborator


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
        with open(log_file, 'r') as file:
            content = file.read()

        pattern = r"Publish memory usage: (\[.*?\])"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            memory_usage_data = match.group(1)
            memory_usage_data = re.sub(r'\S+\.py:\d+', '', memory_usage_data)
            memory_usage_data = memory_usage_data.replace('\n', '').replace(' ', '')
            memory_usage_data = memory_usage_data.replace("'", '"')
            memory_usage_dict = json.loads(memory_usage_data)
            return memory_usage_dict
        else:
            log.error("Memory usage data not found in the log file")
            raise Exception("Memory usage data not found in the log file")
    except Exception as e:
        log.error(f"An error occurred while extracting memory usage: {e}")
        raise e


def write_memory_usage_to_file(memory_usage_dict, output_file):
    """
    Writes memory usage data to a file.
    This function writes the specified memory usage data to the specified output file.
    Args:
        memory_usage_dict (dict): A dictionary containing the memory usage data.
        output_file (str): The path to the output file to which to write the memory usage data.
    """
    try:
        with open(output_file, 'w') as file:
            json.dump(memory_usage_dict, file, indent=4)
    except Exception as e:
        log.error(f"An error occurred while writing memory usage data to file: {e}")
        raise e
