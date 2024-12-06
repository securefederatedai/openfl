# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import concurrent.futures
import logging
import os

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.ssh_helper as sh

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
    # PKI setup for aggregator is done at fixture level
    # Collaborator and model owner operations
    for collaborator in fed_obj.collaborators:
        try:
            log.info(f"Performing operations for {collaborator.collaborator_name}")
            collaborator.generate_sign_request()
            # Below step will add collaborator entries in cols.yaml file of aggregator workspace.
            fed_obj.model_owner.certify_collaborator(collaborator.collaborator_name, collaborator.workspace_path)
            collaborator.import_pki(fed_obj.aggregator.workspace_path)
        except Exception as e:
            log.error(f"Failed to perform PKI setup for {collaborator.collaborator_name}: {e}")
            raise e
    
    # Additional - copy cols.yaml file from aggregator to collaborator workspaces
    # This is for local environment.
    if os.getenv("TEST_ENV") != "docker":
        for collaborator in fed_obj.collaborators:
            try:
                copy_cols_yaml(fed_obj.aggregator.workspace_path, collaborator.workspace_path)
            except Exception as e:
                log.error(f"Failed to copy collaborator yaml file {collaborator.collaborator_name}: {e}")

    success = True

    log.info("PKI setup successfully for all participants")
    return success


def copy_cols_yaml(src_workspace_path, dest_workspace_path):
    """
    Copy cols.yaml file from source workspace to destination workspace
    Args:
        src_workspace_path (str): Source workspace path
        dest_workspace_path (str): Destination workspace path
    """
    src_file = os.path.join(src_workspace_path, "plan", "cols.yaml")
    dest_file = os.path.join(dest_workspace_path, "plan", "cols.yaml")
    cmd = f"cp {src_file} {dest_file}"
    return_code, output, error = sh.run_command(cmd)
    if return_code != 0:
        log.error(f"Failed to copy cols.yaml file: {error}")
        raise Exception(f"Failed to copy cols.yaml file: {error}")
    log.info(f"File cols.yaml copied successfully from {src_workspace_path} to {dest_workspace_path}")


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
        constants.SUCCESS_MARKER not in content and time.time() - start_time < timeout
    ):
        with open(result_file, 'r') as file:
            content = file.read()
        # Print last 2 lines of the log file on screen to track the progress
        log.info(f"{participant.name}: {content.splitlines()[-1:]}")
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
    # Determine the test type based on the markers
    markers = [m.name for m in request.node.iter_markers()]
    os.environ["TEST_ENV"] = test_env = "docker" if "docker" in markers else "task_runner"
    log.info(f"Running the test in {test_env} environment")

    # Validate the model name and create the workspace name
    if not request.config.model_name.upper() in constants.ModelName._member_names_:
        raise ValueError(f"Invalid model name: {request.config.model_name}")

    log.info(
        f"Running federation setup using {test_env} API on single machine with below configurations:\n"
        f"\tNumber of collaborators: {request.config.num_collaborators}\n"
        f"\tNumber of rounds: {request.config.num_rounds}\n"
        f"\tModel name: {request.config.model_name}\n"
        f"\tClient authentication: {request.config.require_client_auth}\n"
        f"\tTLS: {request.config.use_tls}\n"
        f"\tMemory Logs: {request.config.log_memory_usage}"
    )

    # Workspace path by default points to aggregator workspace
    # Collaborators workspace path will be different
    if test_env == "docker":
        # First check if openfl image is available
        dh.check_docker_image()
        # Absolute path is required for docker
        workspace_path = os.path.join("/", request.config.results_dir, request.config.model_name, "aggregator", "workspace")
        col_workspace_path = os.path.join("/", request.config.results_dir, request.config.model_name)
        agg_domain_name = "aggregator"
    else:
        home_dir = os.getenv("HOME")
        workspace_path = os.path.join(home_dir, request.config.results_dir, request.config.model_name, "aggregator", "workspace")
        col_workspace_path = os.path.join(home_dir, request.config.results_dir, request.config.model_name)
        agg_domain_name = "localhost"

    log.info(f"Model owner/aggregator workspace path: {workspace_path}")
    return test_env, request.config.model_name, workspace_path, col_workspace_path, agg_domain_name


def create_persistent_store(participant, results_dir, workspace_template):
    # Create persistent store
    working_directory = os.path.join(os.getenv("HOME"), results_dir)

    cmd_persistent_store = (
        f"export WORKING_DIRECTORY={working_directory}; " \
        f"mkdir -p $WORKING_DIRECTORY/{workspace_template}/{participant.name}/workspace; " \
        "sudo chmod -R 777 $WORKING_DIRECTORY"
    )
    log.info(f"Creating persistent store: {cmd_persistent_store}")
    return_code, output, error = run_command(
        cmd_persistent_store,
        workspace_path=os.getenv("HOME"),
    )
    if error:
        log.error(f"Error in creating persistent store: {error}")
        raise Exception(f"Error in creating persistent store: {error}")

    log.info(f"Persistent store created for {participant.name}")


def run_command(command, workspace_path, error_msg=None, container_id=None, run_in_background=False, bg_file=None, print_output=False):
    """
    Run the command
    Args:
        command (str): Command to run
        work_dir (str): Working directory
        error_msg (str, Optional): Error message
    Returns:
        tuple: Return code, output and error
    """
    return_code, output, error = 0, None, None
    error_msg = error_msg or "Failed to run the command"

    if os.getenv("TEST_ENV") == "docker" and container_id:
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
    if run_in_background:
        bg_file = os.path.join(workspace_path, bg_file)
        log.info(f"\nFile path finally:{bg_file}\n")
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


def modify_plan_for_docker(
    container_name,
    workspace_path,
    new_rounds=None,
    num_collaborators=None,
    disable_client_auth=False,
    disable_tls=False,
    log_memory_usage=False
):
    return


# This functionality is common across multiple participants, thus moved to a common file
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
