# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import concurrent.futures
import logging
import os
import json
import re
import subprocess   # nosec B404
import papermill as pm
from pathlib import Path
import shutil

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.models import collaborator as col_model

log = logging.getLogger(__name__)
home_dir = Path().home()


def setup_pki_for_collaborators(collaborators, model_owner, local_bind_path):
    """
    Setup PKI for trusted communication within the federation

    Args:
        collaborators (list): List of collaborator objects
        model_owner (object): Model owner object
        local_bind_path (str): Local bind path
    Returns:
        bool: True if successful, else False
    """
    # PKI setup for aggregator is done at fixture level
    local_agg_ws_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)

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
            for collaborator in collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception(
                "Failed to generate sign request for one or more collaborators"
            )

    except Exception as e:
        raise e

    # Copy the generated sign request zip from all the collaborators to aggregator
    try:
        results = [
            executor.submit(
                copy_file_between_participants,
                local_src_path=constants.COL_WORKSPACE_PATH.format(
                    local_bind_path, collaborator.name
                ),
                local_dest_path=local_agg_ws_path,
                file_name=f"col_{collaborator.name}_to_agg_cert_request.zip",
            )
            for collaborator in collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception(
                "Failed to copy sign request zip from one or more collaborators to aggregator"
            )
    except Exception as e:
        raise e

    # Certify the collaborator sign requests
    # DO NOT run this in parallel as it causes command to fail with FileNotFoundError for a different collaborator
    for collaborator in collaborators:
        try:
            model_owner.certify_collaborator(
                collaborator_name=collaborator.name,
                zip_name=f"col_{collaborator.name}_to_agg_cert_request.zip",
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
                local_dest_path=constants.COL_WORKSPACE_PATH.format(
                    local_bind_path, collaborator.name
                ),
                file_name=f"agg_to_col_{collaborator.name}_signed_cert.zip",
            )
            for collaborator in collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception(
                "Failed to copy signed certificates from aggregator to one or more collaborators"
            )
    except Exception as e:
        raise e

    return True


def create_tarball_for_collaborators(collaborators, local_bind_path, use_tls, add_data=False):
    """
    Create tarball for all the collaborators
    Args:
        collaborators (list): List of collaborator objects
        local_bind_path (str): Local bind path
        use_tls (bool): Use TLS or not (default is True)
        add_data (bool): Add data to the tarball (default is False)
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    try:

        def _create_tarball(collaborator_name, data_file_path, local_bind_path, add_data):
            """
            Internal function to create tarball for the collaborator.
            If TLS is enabled - include client certificates and signed certificates in the tarball
            If data needs to be added - include the data file in the tarball
            """
            local_col_ws_path = constants.COL_WORKSPACE_PATH.format(
                local_bind_path, collaborator_name
            )
            client_cert_entries = ""
            tarfiles = f"cert_col_{collaborator_name}.tar plan/data.yaml"
            # If TLS is enabled, client certificates and signed certificates are also included
            if use_tls:
                client_cert_entries = [
                    f"cert/client/{f}" for f in os.listdir(f"{local_col_ws_path}/cert/client") if f.endswith(".key")
                ]
                client_certs = " ".join(client_cert_entries) if client_cert_entries else ""
                tarfiles += f" agg_to_col_{collaborator_name}_signed_cert.zip {client_certs}"
                # IMPORTANT: Model XGBoost(xgb_higgs) uses format like data/1 and data/2, thus adding data to tarball in the same format.
                if add_data:
                    tarfiles += f" data/{data_file_path}"

            log.info(f"Tarfile for {collaborator_name} includes: {tarfiles}")
            return_code, output, error = ssh.run_command(
                f"tar -cf {tarfiles}", work_dir=local_col_ws_path
            )
            if return_code != 0:
                raise Exception(
                    f"Failed to create tarball for {collaborator_name}: {error}"
                )
            return True

        results = [
            executor.submit(
                _create_tarball, collaborator.name, data_file_path=index, local_bind_path=local_bind_path, add_data=add_data
            )
            for index, collaborator in enumerate(collaborators, start=1)
        ]
        if not all([f.result() for f in results]):
            raise Exception("Failed to create tarball for one or more collaborators")
    except Exception as e:
        raise e

    return True


def import_pki_for_collaborators(collaborators, local_bind_path):
    """
    Import and certify the CSR for the collaborators
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    local_agg_ws_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)
    try:
        results = [
            executor.submit(
                collaborator.import_pki,
                zip_name=f"agg_to_col_{collaborator.name}_signed_cert.zip",
            )
            for collaborator in collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception(
                "Failed to import and certify the CSR for one or more collaborators"
            )
    except Exception as e:
        raise e

    # Copy the cols.yaml file from aggregator to all the collaborators
    # File cols.yaml is updated after PKI setup
    try:
        results = [
            executor.submit(
                copy_file_between_participants,
                local_src_path=os.path.join(local_agg_ws_path, "plan"),
                local_dest_path=constants.COL_PLAN_PATH.format(
                    local_bind_path, collaborator.name
                ),
                file_name="cols.yaml",
                run_with_sudo=True,
            )
            for collaborator in collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception(
                "Failed to copy cols.yaml file from aggregator to one or more collaborators"
            )
    except Exception as e:
        raise e

    return True


def copy_file_between_participants(
    local_src_path, local_dest_path, file_name, run_with_sudo=False
):
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
    return_code, output, error = ssh.run_command(cmd)
    if return_code != 0:
        log.error(f"Failed to copy file: {error}")
        raise Exception(f"Failed to copy file: {error}")
    log.info(
        f"File {file_name} copied successfully from {local_src_path} to {local_dest_path}"
    )
    return True


def run_federation(fed_obj, install_dependencies=True, with_docker=False):
    """
    Start the federation
    Args:
        fed_obj (object): Federation fixture object
        install_dependencies (bool): Install dependencies on collaborators (default is True)
        with_docker (bool): Flag specific to dockerized workspace scenario. Default is False.
    Returns:
        list: List of response files for all the participants
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    if install_dependencies:
        install_dependencies_on_collaborators(fed_obj)

    # As the collaborators will wait for aggregator to start, we need to start them in parallel.
    futures = [
        executor.submit(
            participant.start,
            constants.AGG_COL_RESULT_FILE.format(
                fed_obj.workspace_path, participant.name
            ),
            with_docker=with_docker,
        )
        for participant in fed_obj.collaborators + [fed_obj.aggregator]
    ]

    # Result will contain response files for all the participants.
    results = [f.result() for f in futures]
    if not all(results):
        raise Exception("Failed to start one or more participants")
    return results


def run_federation_for_dws(fed_obj, use_tls):
    """
    Start the federation
    Args:
        fed_obj (object): Federation fixture object
        use_tls (bool): Use TLS or not (default is True)
    Returns:
        list: List of response files for all the participants
    """
    executor = concurrent.futures.ThreadPoolExecutor()

    try:
        results = [
            executor.submit(
                run_command,
                command=f"tar -xf /workspace/certs.tar",
                workspace_path="",
                error_msg=f"Failed to extract certificates for {participant.name}",
                container_id=participant.container_id,
                with_docker=True,
            )
            for participant in [fed_obj.aggregator] + fed_obj.collaborators
        ]
        if not all([f.result() for f in results]):
            raise Exception(
                "Failed to extract certificates for one or more participants"
            )
    except Exception as e:
        raise e

    if use_tls:
        try:
            results = [
                executor.submit(
                    collaborator.import_pki,
                    zip_name=f"agg_to_col_{collaborator.name}_signed_cert.zip",
                    with_docker=True,
                )
                for collaborator in fed_obj.collaborators
            ]
            if not all([f.result() for f in results]):
                raise Exception(
                    "Failed to import and certify the CSR for one or more collaborators"
                )
        except Exception as e:
            raise e

    # Start federation run for all the participants
    return run_federation(fed_obj, with_docker=True)


def install_dependencies_on_collaborators(fed_obj):
    """
    Install dependencies on all the collaborators
    """
    executor = concurrent.futures.ThreadPoolExecutor()
    # Install dependencies on collaborators
    # This is a time taking process, thus doing at this stage after all verification is done
    log.info("Installing dependencies on collaborators. This might take some time...")
    futures = [
        executor.submit(participant.install_dependencies)
        for participant in fed_obj.collaborators
    ]
    results = [f.result() for f in futures]
    log.info(
        f"Results from all the collaborators for installation of dependencies: {results}"
    )

    if not all(results):
        raise Exception("Failed to install dependencies on one or more collaborators")


def verify_federation_run_completion(fed_obj, results, test_env, num_rounds):
    """
    Verify the completion of the process for all the participants
    Args:
        fed_obj (object): Federation fixture object
        results (list): List of results
        test_env (str): Test environment
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
            test_env,
            local_bind_path=fed_obj.local_bind_path,
        )
        for i, participant in enumerate(fed_obj.collaborators + [fed_obj.aggregator])
    ]

    # Result will contain a list of boolean values for all the participants.
    # True - successful completion, False - failed/incomplete
    results = [f.result() for f in futures]
    log.info(f"Results from all the participants: {results}")

    # If any of the participant failed, return False, else return True
    return all(results)


def _verify_completion_for_participant(
    participant, num_rounds, result_file, test_env, time_for_each_round=100, local_bind_path=None
):
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
    time.sleep(20)  # Wait for some time before checking the log file
    # Set timeout based on the number of rounds and time for each round
    timeout = 600 + (time_for_each_round * num_rounds)  # in seconds

    # In case of docker environment, get the logs from local path which is mounted to the container
    if test_env == "task_runner_dockerized_ws":
        result_file = constants.AGG_COL_RESULT_FILE.format(
            local_bind_path, participant.name
        )
        ssh.copy_file_from_docker(
            participant.name, f"/workspace/{participant.name}.log", result_file
        )

    log.info(f"Result file is: {result_file}")

    # Do not open file here as it will be opened in the loop below
    # Also it takes time for the federation run to start and write the logs
    content = [""]

    start_time = time.time()
    while (
        constants.SUCCESS_MARKER not in content and time.time() - start_time < timeout
    ):
        with open(result_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]
        content = list(filter(str.rstrip, lines))[-1:]

        # Print last line of the log file on screen to track the progress
        log.info(f"Last line in {participant.name} log: {content}")
        if constants.SUCCESS_MARKER in content:
            break
        log.info(f"Process is yet to complete for {participant.name}")
        time.sleep(45)

        # Copy the log file from docker container to local machine everytime to get the latest logs
        if test_env == "task_runner_dockerized_ws":
            ssh.copy_file_from_docker(
                participant.name,
                f"/workspace/{participant.name}.log",
                constants.AGG_COL_RESULT_FILE.format(local_bind_path, participant.name),
            )

    if constants.SUCCESS_MARKER not in content:
        log.error(
            f"Process failed/is incomplete for {participant.name} after timeout of {timeout} seconds"
        )
        return False
    else:
        log.info(
            f"Process completed for {participant.name} in {time.time() - start_time} seconds"
        )
        return True


def federation_env_setup_and_validate(request):
    """
    Setup the federation environment and validate the configurations
    Args:
        request (object): Request object
    Returns:
        tuple: Model name, workspace path, local bind path, aggregator domain name
    """
    agg_domain_name = "localhost"

    # Determine the test type based on the markers
    test_env = request.config.test_env

    # Validate the model name and create the workspace name
    if not request.config.model_name.upper() in constants.ModelName._member_names_:
        raise ValueError(f"Invalid model name: {request.config.model_name}")

    # Set the workspace path
    home_dir = Path().home()
    local_bind_path = os.path.join(
        home_dir, request.config.results_dir, request.config.model_name
    )
    workspace_path = local_bind_path

    if test_env == "task_runner_dockerized_ws":

        agg_domain_name = "aggregator"
        # Cleanup docker containers
        dh.cleanup_docker_containers()
        dh.remove_docker_network()
        dh.create_docker_network()

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
    return request.config.model_name, workspace_path, local_bind_path, agg_domain_name


def add_local_workspace_permission(local_bind_path):
    """
    Add permission to workspace. This is aggregator/model owner specific operation.
    Args:
        workspace_path (str): Workspace path
        agg_container_id (str): Container ID
    """
    try:
        agg_workspace_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)
        return_code, output, error = run_command(
            f"sudo chmod -R 777 {agg_workspace_path}",
            workspace_path=local_bind_path,
        )
        if return_code != 0:
            raise Exception(f"Failed to add local permission to workspace: {error}")

        log.debug(
            f"Recursive permission added to workspace on local machine: {agg_workspace_path}"
        )
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
            f"export WORKING_DIRECTORY={local_bind_path}; "
            f"mkdir -p $WORKING_DIRECTORY/{participant_name}/workspace; "
            "sudo chmod -R 755 $WORKING_DIRECTORY"
        )
        log.debug(f"Creating persistent store")
        return_code, output, error = run_command(
            cmd_persistent_store,
            workspace_path=Path().home(),
        )
        if error:
            raise ex.PersistentStoreCreationException(f"{error_msg}: {error}")

        log.info(f"Persistent store created for {participant_name}")

    except Exception as e:
        raise ex.PersistentStoreCreationException(f"{error_msg}: {e}")


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
        bg_file = open(bg_file, "w", buffering=1)
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
    msg_received = [line for line in output if constants.SUCCESS_MARKER in line]
    log.info(f"Message received: {msg_received}")
    if return_code == 0 and len(msg_received):
        log.info(success_msg)
    else:
        log.error(f"{error_msg}: {error}")
        if raise_exception:
            raise Exception(f"{error_msg}: {error}")


def setup_collaborator(index, workspace_path, local_bind_path):
    """
    Setup the collaborator
    Includes - creation of collaborator objects, starting docker container, importing workspace, creating collaborator
    Args:
        index (int): Index of the collaborator. Starts with 1.
        workspace_path (str): Workspace path
        local_bind_path (str): Local bind path
    """
    local_agg_ws_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)

    try:
        collaborator = col_model.Collaborator(
            collaborator_name=f"collaborator{index}",
            data_directory_path=index,
            workspace_path=f"{workspace_path}/collaborator{index}/workspace",
        )
        create_persistent_store(collaborator.name, local_bind_path)

    except Exception as e:
        raise ex.PersistentStoreCreationException(
            f"Failed to create persistent store for {collaborator.name}: {e}"
        )

    try:
        local_col_ws_path = constants.COL_WORKSPACE_PATH.format(
            local_bind_path, collaborator.name
        )
        copy_file_between_participants(
            local_agg_ws_path, local_col_ws_path, constants.AGG_WORKSPACE_ZIP_NAME
        )
        collaborator.import_workspace()
    except Exception as e:
        raise ex.WorkspaceImportException(
            f"Failed to import workspace for {collaborator.name}: {e}"
        )

    try:
        collaborator.create_collaborator()
    except Exception as e:
        raise ex.CollaboratorCreationException(f"Failed to create collaborator: {e}")

    return collaborator


def setup_collaborator_data(collaborators, model_name, local_bind_path):
    """
    Function to setup the data for collaborators.
    IMP: This function is specific to the model and should be updated as per the model requirements.
    Args:
        collaborators (list): List of collaborator objects
        model_name (str): Model name
        local_bind_path (str): Local bind path
    """
    # Check if data already exists, if yes, skip the download part
    # This is mainly helpful in case of re-runs
    if all(os.path.exists(os.path.join(collaborator.workspace_path, "data", str(index))) for index, collaborator in enumerate(collaborators, start=1)):
        log.info("Data already exists for all the collaborators. Skipping the download part..")
        return
    else:
        log.info("Data does not exist for all the collaborators. Proceeding with the download..")
        # Below step will also modify the data.yaml file for all the collaborators
        download_data(collaborators, model_name, local_bind_path)

    log.info("Data setup is complete for all the collaborators")


def download_data(collaborators, model_name, local_bind_path):
    """
    Download the data for the model and copy to the respective collaborator workspaces
    Also modify the data.yaml file for all the collaborators
    Args:
        collaborators (list): List of collaborator objects
        model_name (str): Model name
        local_bind_path (str): Local bind path
    Returns:
        bool: True if successful, else False
    """
    log.info(f"Copying {constants.DATA_SETUP_FILE} from one of the collaborator workspaces to the local bind path..")
    try:
        shutil.copyfile(
            src=os.path.join(collaborators[0].workspace_path, "src", constants.DATA_SETUP_FILE),
            dst=os.path.join(local_bind_path, constants.DATA_SETUP_FILE)
        )
    except Exception as e:
        raise ex.DataSetupException(f"Failed to copy data setup file: {e}")

    log.info("Downloading the data for the model. This will take some time to complete based on the data size ..")
    try:
        command = ["python", constants.DATA_SETUP_FILE, str(len(collaborators))]
        subprocess.run(command, cwd=local_bind_path, check=True)
    except Exception:
        raise ex.DataSetupException(f"Failed to download data for {model_name}")

    try:
        # Copy the data to the respective workspaces based on the index
        for index, collaborator in enumerate(collaborators, start=1):
            src_folder = os.path.join(local_bind_path, "data", str(index))
            dst_folder = os.path.join(collaborator.workspace_path, "data", str(index))
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                log.info(f"Copied data from {src_folder} to {dst_folder}")
            else:
                raise ex.DataSetupException(f"Source folder {src_folder} does not exist for {collaborator.name}")

            # Modify the data.yaml file for all the collaborators
            collaborator.modify_data_file(
                constants.COL_DATA_FILE.format(local_bind_path, collaborator.name),
                index,
            )
    except Exception as e:
        raise ex.DataSetupException(f"Failed to modify the data file: {e}")

    # Below step is specific to XGBoost model which uses higgs_data folder to create data folders.
    shutil.rmtree(os.path.join(local_bind_path, "higgs_data"), ignore_errors=True)

    return True


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


def start_docker_containers_for_dws(
    participants, workspace_path, local_bind_path, image_name
):
    """
    Start docker containers for the participants
    Args:
        participants (list): List of participant objects (collaborators and aggregator)
        workspace_path (str): Workspace path
        local_bind_path (str): Local bind path
        image_name (str): Docker image name
    """
    for participant in participants:
        try:
            if participant.name == "aggregator":
                local_ws_path = f"{local_bind_path}/aggregator/workspace"
                local_cert_tar = "cert_agg.tar"
            else:
                local_ws_path = f"{local_bind_path}/{participant.name}/workspace"
                local_cert_tar = f"cert_col_{participant.name}.tar"

            # In case of dockerized workspace, the workspace gets created inside folder with image name
            container = dh.start_docker_container(
                container_name=participant.name,
                workspace_path=workspace_path,
                local_bind_path=local_bind_path,
                image=image_name,
                mount_mapping=[
                    f"{local_ws_path}/{local_cert_tar}:/{image_name}/certs.tar"
                ],
            )
            participant.container_id = container.id
        except Exception as e:
            raise ex.DockerException(
                f"Failed to start {participant.name} docker environment: {e}"
            )


def start_director(workspace_path, dir_res_file):
    """
    Start the director.
    Args:
        workspace_path (str): Workspace path
        dir_res_file (str): Director result file
    Returns:
        bool: True if successful, else False
    """
    try:
        error_msg = "Failed to start the director"
        return_code, output, error = run_command(
            "./start_director.sh",
            error_msg=error_msg,
            workspace_path=os.path.join(workspace_path, "director"),
            run_in_background=True,
            bg_file=dir_res_file,
        )
        log.debug(f"Director start: Return code: {return_code}, Output: {output}, Error: {error}")
        log.info(
            "Waiting for 30s for the director to start. With no retry mechanism in place, "
            "envoys will fail immediately if the director is not ready."
        )
        time.sleep(30)
    except ex.DirectorStartException as e:
        raise e
    return True


def start_envoy(envoy_name, workspace_path, res_file):
    """
    Start given envoy.
    Args:
        envoy_name (str): Name of the envoy. For e.g. Bangalore, Chandler (case sensitive)
        workspace_path (str): Workspace path
        res_file (str): Result file to track the logs.
    Returns:
        bool: True if successful, else False
    """
    try:
        error_msg = f"Failed to start {envoy_name} envoy"
        return_code, output, error = run_command(
            f"./start_envoy.sh {envoy_name} {envoy_name}_config.yaml",
            error_msg=error_msg,
            workspace_path=os.path.join(workspace_path, envoy_name),
            run_in_background=True,
            bg_file=res_file,
        )
        log.debug(f"{envoy_name} start: Return code: {return_code}, Output: {output}, Error: {error}")
    except ex.EnvoyStartException as e:
        raise e
    return True


def create_federated_runtime_participant_res_files(results_dir, envoys, model_name="301_mnist_watermarking"):
    """
    Create result log files for the director and envoys.
    Args:
        results_dir (str): Results directory
        envoys (list): List of envoys
        model_name (str): Model name
    Returns:
        tuple: Result path and participant result files (including director)
    """
    participant_res_files = {}
    result_path = os.path.join(
        home_dir, results_dir, model_name
    )
    os.makedirs(result_path, exist_ok=True)

    for participant in envoys + ["director"]:
        res_file = os.path.join(result_path, f"{participant.lower()}.log")
        participant_res_files[participant.lower()] = res_file
        # Create the file
        open(res_file, 'w').close()


    return result_path, participant_res_files


def check_envoys_director_conn_federated_runtime(
    notebook_path, expected_envoys, director_node_fqdn="localhost", director_port=50050
):
    """
    Function to check if the envoys are connected to the director for Federated Runtime notebooks.
    Args:
        notebook_path (str): Path to the notebook
        expected_envoys (list): List of expected envoys
        director_node_fqdn (str): Director node FQDN
        director_port (int): Director port
    Returns:
        bool: True if all the envoys are connected to the director, else False
    """
    from openfl.experimental.workflow.runtime import FederatedRuntime

    # Number of retries and delay between retries in seconds
    MAX_RETRIES = RETRY_DELAY = 5

    federated_runtime = FederatedRuntime(
        collaborators=expected_envoys,
        director={
            "director_node_fqdn": director_node_fqdn,
            "director_port": director_port,
        },
        notebook_path=notebook_path,
    )
    # Retry logic
    for attempt in range(MAX_RETRIES):
        actual_envoys = federated_runtime.get_envoys()
        if all(
            sorted(expected_envoys) == sorted(actual_envoys)
            for expected_envoys, actual_envoys in [(expected_envoys, actual_envoys)]
        ):
            log.info("All the envoys are connected to the director")
            return True
        else:
            log.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES}: Not all envoys are connected. Retrying in {RETRY_DELAY} seconds..."
            )
            time.sleep(RETRY_DELAY)

    return False


def run_notebook(notebook_path, output_notebook_path):
    """
    Function to run the notebook.
    Args:
        notebook_path (str): Path to the notebook
        participant_res_files (dict): Dictionary containing participant names and their result log files
    Returns:
        bool: True if successful, else False
    """
    try:
        log.info(f"Running the notebook: {notebook_path} with output notebook path: {output_notebook_path}")
        output = pm.execute_notebook(
            input_path=notebook_path,
            output_path=output_notebook_path,
            request_save_on_cell_execute=True,
            autosave_cell_every=5, # autosave every 5 seconds
            log_output=True,
        )
    except pm.exceptions.PapermillExecutionError as e:
        log.error(f"PapermillExecutionError: {e}")
        raise e

    except ex.NotebookRunException as e:
        log.error(f"Failed to run the notebook: {e}")
        raise e
    return True


def verify_federated_runtime_experiment_completion(participant_res_files):
    """
    Verify the completion of the experiment using the participant logs.
    Args:
        participant_res_files (dict): Dictionary containing participant names and their result log files
    Returns:
        bool: True if successful, else False
    """
    # Check participant logs for successful completion
    for name, result_file in participant_res_files.items():
        # Do not open file here as it will be opened in the loop below
        # Also it takes time for the federation run to start and write the logs
        with open(result_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]
        last_7_lines = list(filter(str.rstrip, lines))[-7:]
        if (
            name == "director"
            and [1 for content in last_7_lines if "Experiment FederatedFlow_MNIST_Watermarking was finished successfully" in content]
        ):
            log.debug(f"Process completed for {name}")
            continue
        elif name != "director" and [1 for content in last_7_lines if "End of Federation reached." in content]:
            log.debug(f"Process completed for {name}")
            continue
        else:
            log.error(f"Process failed for {name}")
            return False
    return True
