# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import concurrent.futures
import logging
import yaml
import os
import subprocess   # nosec B404
from pathlib import Path

import tests.end_to_end.utils.defaults as defaults
import tests.end_to_end.utils.db_helper as db_helper
import tests.end_to_end.utils.docker_helper as dh
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.helper as helper
import tests.end_to_end.utils.interruption_helper as intr_helper
import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.models import collaborator as col_model
from tests.end_to_end.utils.generate_report import convert_to_json

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
    local_agg_ws_path = defaults.AGG_WORKSPACE_PATH.format(local_bind_path)

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
                local_src_path=defaults.COL_WORKSPACE_PATH.format(
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
                local_dest_path=defaults.COL_WORKSPACE_PATH.format(
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
            local_col_ws_path = defaults.COL_WORKSPACE_PATH.format(
                local_bind_path, collaborator_name
            )
            client_cert_entries = ""
            tarfiles = f"cert_{collaborator_name}.tar plan/data.yaml"
            # If TLS is enabled, client certificates and signed certificates are also included
            if use_tls:
                client_cert_entries = [
                    f"cert/client/{f}" for f in os.listdir(f"{local_col_ws_path}/cert/client") if f.endswith(".key")
                ]
                client_certs = " ".join(client_cert_entries) if client_cert_entries else ""
                tarfiles += f" agg_to_col_{collaborator_name}_signed_cert.zip {client_certs}"
                # IMPORTANT: Models XGBoost(xgb_higgs) and Flower use format like data/1 and data/2, thus adding data to tarball in the same format.
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


def import_pki_for_collaborators(collaborators):
    """
    Import and certify the CSR for the collaborators
    """
    executor = concurrent.futures.ThreadPoolExecutor()
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


def _check_aggregator_protocol_log(aggregator):
    """
    Check if the aggregator started with the correct protocol by inspecting its log file.
    Args:
        aggregator (object): Aggregator object with res_file and transport_protocol attributes.
    Raises:
        Exception: If the expected protocol start message is not found in the logs.
    """
    start_time = time.time()
    found = False
    while time.time() - start_time < 30:
        with open(aggregator.res_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]
        last_lines = lines[-5:]
        if aggregator.transport_protocol == defaults.TransportProtocol.REST.value:
            expected_msg = defaults.AGGREGATOR_REST_CLIENT
        else:
            expected_msg = defaults.AGGREGATOR_gRPC_CLIENT

        msg_received = [line for line in last_lines if expected_msg.lower() in line.lower()]
        if msg_received:
            found = True
            break
        time.sleep(10)
    if not found:
        raise Exception(
            f"Aggregator did not start with {aggregator.transport_protocol} protocol. Check the logs for more details"
        )
    log.info(f"Aggregator started with {aggregator.transport_protocol} protocol")


def run_federation(fed_obj):
    """
    Start the federation
    Args:
        fed_obj (object): Federation fixture object
    Returns:
        bool: True if successful, else False
    """

    # Set the backend (KERAS_BACKEND) for Keras as an environment variable
    if "keras" in fed_obj.model_name:
        _ = set_keras_backend(fed_obj.model_name)

    # Start the aggregator
    start_aggregator(fed_obj)

    for participant in fed_obj.collaborators:
        try:
            participant.start()
        except Exception as e:
            log.error(f"Failed to start {participant.name}: {e}")
            raise e
    return True


def run_federation_for_dws(fed_obj, use_tls):
    """
    Start the federation
    Args:
        fed_obj (object): Federation fixture object
        use_tls (bool): Use TLS or not (default is True)
    Returns:
        bool: True if successful, else False
    """
    for participant in [fed_obj.aggregator] + fed_obj.collaborators:
        try:
            container = dh.start_docker_container_with_federation_run(
                participant=participant,
                image=defaults.DFLT_WORKSPACE_NAME,
                use_tls=use_tls,
                env_keyval_list=set_keras_backend(fed_obj.model_name) if "keras" in fed_obj.model_name else None,
            )
        except Exception as e:
            log.error(f"Failed to start docker container for {participant.name}: {e}")
            raise e

        participant.container_id = container.id
        participant.res_file = os.path.join(participant.workspace_path, "logs", f"{participant.name}.log")

    return True


def verify_federation_run_completion(fed_obj, test_env, num_rounds, time_for_each_round=100):
    """
    Verify the completion of the process for all the participants
    Args:
        fed_obj (object): Federation fixture object
        test_env (str): Test environment
        num_rounds (int): Number of rounds
        time_for_each_round (int): Time for each round (in seconds)
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
            num_collaborators=len(fed_obj.collaborators),
            time_for_each_round=time_for_each_round,
        )
        for participant in fed_obj.collaborators + [fed_obj.aggregator]
    ]

    # Result will contain a list of boolean values for all the participants.
    # True - successful completion, False - failed/incomplete
    results = [f.result() for f in futures]
    log.debug(f"Results from all the participants: {results}")

    # If any of the participant failed, return False, else return True
    return all(results)


def _verify_completion_for_participant(
    participant, num_rounds, num_collaborators, time_for_each_round=100
):
    """
    Verify the completion of the process for the participant
    Args:
        participant (object): Participant object
        num_rounds (int): Number of rounds
        num_collaborators (int): Number of collaborators
        time_for_each_round (int): Time for each round
    Returns:
        bool: True if successful, else False
    """
    start_time = time.time()
    # Wait for a min so that log files are available
    while not os.path.exists(participant.res_file):
        if time.time() - start_time > 60:
            raise Exception(f"Log file {participant.res_file} not found after 60 seconds")
        time.sleep(10)

    # Set timeout based on the number of rounds and time for each round
    timeout = 600 + (time_for_each_round * num_rounds)  # in seconds

    # Do not open file here as it will be opened in the loop below
    # Also it takes time for the federation run to start and write the logs
    content = [""]

    while time.time() - start_time < timeout:
        with open(participant.res_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]

        # Get the desired no of lines from the log file
        if num_collaborators < 5:
            reverse_index = 10
        else:
            # For more than 5 collaborators, set the index to 10 + number of collaborators
            # This is to ensure that we get the completion message for all the collaborators
            reverse_index = num_collaborators + 5

        # Get the required lines from the log file
        if len(lines) >= reverse_index:
            content = lines[-reverse_index:]
        else:
            content = lines

        # Print last line of the log file on screen to track the progress
        log.info(f"Last line in {participant.name} log: {lines[-1:]}")

        # If in logs Exception is encountered, throw Exception and stop the process
        if defaults.EXCEPTION in content:
            log.error(
                f"Process {participant.name} is throwing Exception. Check the logs for more details"
            )
            raise Exception(f"Process failed for {participant.name}")

        msg_received = [line for line in content if defaults.AGG_END_MSG in line or defaults.COL_END_MSG in line]
        if msg_received:
            log.info(f"Process completed for {participant.name}")
            break

        # If process.poll() has a value, it means the process has completed
        # If None, it means the process is still running
        # This is applicable for native process only
        if participant.start_process:
            if participant.start_process.poll() or not len(intr_helper.get_pids_for_active_command(participant.name)):
                log.info(f"No processes found for participant {participant.name}")
                break
            else:
                log.info(f"Process is yet to complete for {participant.name}")
        else:
            # Dockerized workspace scenario
            log.info(f"No process found for participant {participant.name}")

        time.sleep(45)

    # Read tensor.db file for aggregator to check if the process is completed
    if participant.name == "aggregator" and num_rounds > 1:
        current_round = get_current_round(participant.tensor_db_file)
        if (current_round + 1) != num_rounds:
            raise Exception(f"Process completed but only till round {current_round}")

    return True


def federation_env_setup_and_validate(request, eval_scope=False):
    """
    Setup the federation environment and validate the configurations
    Args:
        request (object): Request object
        eval_scope (bool): If True, sets up the evaluation scope for a single round
    Returns:
        tuple: Model name, workspace path, local bind path, aggregator domain name
    """
    agg_domain_name = "localhost"

    # Determine the test type based on the markers
    test_env = request.config.test_env

    # Validate the model name and create the workspace name
    if not request.config.model_name.replace("/", "_").replace("-", "_").upper() in defaults.ModelName._member_names_:
        raise ValueError(f"Invalid model name: {request.config.model_name}")

    # Set the workspace path specific to the model and the test case
    home_dir = Path().home()
    local_bind_path = os.path.join(
        home_dir, request.config.results_dir, request.node.name, request.config.model_name.replace("/", "_")
    )

    num_rounds = request.config.num_rounds

    if eval_scope:
        local_bind_path = f"{local_bind_path}_eval"
        log.info(f"Running evaluation for the model: {request.config.model_name}")

    workspace_path = local_bind_path

    # if path exists delete it
    if os.path.exists(workspace_path):
        remove_workspace(workspace_path)

    if test_env == "task_runner_dockerized_ws":
        agg_domain_name = "aggregator"
        # Cleanup docker containers
        dh.cleanup_docker_containers()
        dh.remove_docker_network()
        dh.create_docker_network()

    request.config.transport_protocol = defaults.TransportProtocol.REST.value if request.config.tr_rest_protocol else defaults.TransportProtocol.GRPC.value
    log.info(
        f"Running federation setup using {test_env} API on single machine with below configurations:\n"
        f"Number of collaborators: {request.config.num_collaborators}\n"
        f"Number of rounds: {num_rounds}\n"
        f"Model name: {request.config.model_name}\n"
        f"Client authentication: {request.config.require_client_auth}\n"
        f"TLS: {request.config.use_tls}\n"
        f"Secure Aggregation: {request.config.secure_agg}\n"
        f"Transport protocol: {request.config.transport_protocol}\n"
        f"Memory Logs: {request.config.log_memory_usage}\n"
        f"Results directory: {request.config.results_dir}\n"
        f"Workspace path: {workspace_path}"
    )
    return workspace_path, local_bind_path, agg_domain_name


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
            f"mkdir -p $WORKING_DIRECTORY/{participant_name}/workspace"
        )
        log.debug(f"Creating persistent store")
        return_code, output, error = helper.run_command(
            cmd_persistent_store,
            workspace_path=Path().home(),
        )
        if error:
            raise ex.PersistentStoreCreationException(f"{error_msg}: {error}")

        log.info(f"Persistent store created for {participant_name}")

    except Exception as e:
        raise ex.PersistentStoreCreationException(f"{error_msg}: {e}")


def setup_collaborator(index, workspace_path, local_bind_path, data_path=None, calc_hash=False, colab_bucket_mapping=None, transport_protocol="grpc"):
    """
    Setup the collaborator
    Includes - creation of collaborator objects, starting docker container, importing workspace, creating collaborator
    Args:
        index (int): Index of the collaborator. Starts with 1.
        workspace_path (str): Workspace path
        local_bind_path (str): Local bind path
        data_path (str): Data path
        calc_hash (bool): Flag to indicate if hash calculation is required
        colab_bucket_mapping (dict): Mapping of collaborator and its datasources
        transport_protocol (str): Transport protocol (default: "gRPC")
    """
    local_agg_ws_path = defaults.AGG_WORKSPACE_PATH.format(local_bind_path)

    # If datasource path exists, it indicates that the collaborator is using a custom data source
    # After importing workspace, copy the datasources.json file to the collaborator workspace/data directory
    # and set the data_directory_path to "data"
    datasource_path = os.path.join(str(Path(local_bind_path).parents[1]), "datasources", f"collaborator{index}")
    if not os.path.exists(datasource_path):
        datasource_path = None

    try:
        collaborator = col_model.Collaborator(
            collaborator_name=f"collaborator{index}",
            transport_protocol=transport_protocol,
            data_directory_path=index if datasource_path is None else "data",
            workspace_path=f"{workspace_path}/collaborator{index}/workspace",
        )
        create_persistent_store(collaborator.name, local_bind_path)

    except Exception as e:
        raise ex.PersistentStoreCreationException(
            f"Failed to create persistent store for {collaborator.name}: {e}"
        )

    try:
        local_col_ws_path = defaults.COL_WORKSPACE_PATH.format(
            local_bind_path, collaborator.name
        )
        copy_file_between_participants(
            local_agg_ws_path, local_col_ws_path, f"{defaults.DFLT_WORKSPACE_NAME}.zip"
        )
        collaborator.import_workspace()
    except Exception as e:
        raise ex.WorkspaceImportException(
            f"Failed to import workspace for {collaborator.name}: {e}"
        )

    # If datasources path exist, copy the data files to the collaborator workspace
    if datasource_path:
        try:
            copy_file_between_participants(
                local_src_path=datasource_path,
                local_dest_path=os.path.join(collaborator.workspace_path, "data"),
                file_name="datasources.json",
                run_with_sudo=True,
            )
        except Exception as e:
            raise ex.DataCopyException(
                f"Failed to copy datasources.json for {collaborator.name}: {e}"
            )

    try:
        collaborator.create_collaborator()
    except Exception as e:
        raise ex.CollaboratorCreationException(f"Failed to create collaborator: {e}")

    # Calculate the hash of collaborator datasource (specific to torch/histology_s3 model).
    if datasource_path:
        try:
            # Calculate hash for the collaborator
            collaborator.calculate_hash()
        except Exception as e:
            raise ex.HashCalculationException(
                f"Failed to calculate hash for {collaborator.name}: {e}"
            )

    return collaborator


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
        return_code, output, error = helper.run_command(
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
        return_code, output, error = helper.run_command(
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


def create_federated_runtime_participant_res_files(results_dir, envoys, model_name):
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
            and [1 for content in last_7_lines if "was finished successfully" in content]
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


def get_current_round(database_file: str) -> int:
    """
    Get the current round number from the database file
    Args:
        database_file (str): Database file
    Returns:
        int: Current round number
    """
    return int(db_helper.get_key_value_from_db("round_number", database_file))


def get_best_agg_score(database_file=None, agg_metric_file=None, max_retries=10, sleep_interval=5):
    """
    Get the best aggregated score from the database file or aggregator metrics file
    Args:
        database_file (str): Database file. Optional.
        agg_metric_file (str): Aggregator metrics file. Optional.
        max_retries (int): Maximum number of retries to get the best score in case of database_file. Default is 10.
        sleep_interval (int): Sleep interval between retries in seconds in case of database_file. Default is 5 seconds.
    Returns:
        float: Best aggregated score
    """
    # If both the params are not present, raise exception
    if not database_file and not agg_metric_file:
        raise ValueError("Either database_file or agg_metric_file should be provided")

    if database_file:
        return db_helper.get_key_value_from_db("best_score", database_file, max_retries=max_retries, sleep_interval=sleep_interval)
    else:
        json_file = convert_to_json(agg_metric_file)
        best_score = json_file[-1].get(defaults.AGG_METRIC_MODEL_ACCURACY_KEY)
        if best_score:
            return float(best_score)
        else:
            raise ValueError("Best score not found in the aggregator metrics file")


def validate_round_increment(inp_round, database_file, total_rounds, timeout=300, sleep_interval=5):
    """
    Validate if the round number has increased from inp_round by fetching the value via get_key_value_from_db
    and retrying with some wait time for input timeout.
    Args:
        inp_round (int): The initial round number to compare against.
        database_file (str): The path to the database file.
        total_rounds (int): The total number of rounds expected.
        timeout (int): The maximum time to wait in seconds.
            Default is 300 seconds as some of the models take more time to complete the round.
        sleep_interval (int): The wait time between retries in seconds. Default is 5 seconds.
    Returns:
        round number(int) if current round number has increased, else False.
    """
    if inp_round == total_rounds:
        log.info("Federation is already at the last round.")
        return inp_round

    start_time = time.time()
    while time.time() - start_time < timeout:
        current_round = get_current_round(database_file)
        # Sometimes round number is not updated immediately, thus checking for current_round > inp_round + 1
        if current_round > inp_round + 1:
            log.info(f"Round number has increased from {inp_round} to {current_round}")
            return current_round
        # Check if already at the final round (round no. index starts with 0)
        if current_round + 1 == total_rounds:
            log.info(f"Already at the final round")
            return current_round
        log.info(f"Round number has not increased from {inp_round}. Retrying in {sleep_interval} seconds...")
        time.sleep(sleep_interval)
    log.warning(f"Round number has not increased from {inp_round} after {timeout} seconds")
    return False


def set_keras_backend(model_name):
    """
    Function to set the KERAS_BACKEND environment variable based on the model name.
    Args:
        model_name (str): Model name
    Returns:
        list: List of environment variables
    """
    if "keras" not in model_name:
        return None

    parts = model_name.split("/")
    # TODO - modify the logic if the model name changes to have more than 3 parts
    if len(parts) == 3:
        backend = parts[1]
    else:
        return None

    os.environ["KERAS_BACKEND"] = backend

    return [f"KERAS_BACKEND={backend}"]


def remove_workspace(path):
    """
    Recursively delete given workspace and its contents, including symbolic links.

    Args:
        path (str): The path to the workspace to be deleted.
    """
    if os.path.islink(path) or os.path.isfile(path):
        subprocess.run(['sudo', 'rm', '-f', path], check=True)
    elif os.path.isdir(path):
        for entry in os.scandir(path):
            remove_workspace(entry.path)
        subprocess.run(['sudo', 'rmdir', path], check=True)


def get_agg_addr_port(plan_file):
    """
    Get the aggregator address and port
    Returns:
        tuple: Aggregator address and port
    """
    try:
        with open(plan_file) as fp:
            data = yaml.safe_load(fp)

        agg_addr = data["network"]["settings"]["agg_addr"]
        agg_port = data["network"]["settings"]["agg_port"]
        return agg_addr, agg_port
    except Exception as e:
        raise ex.PlanReadException(f"Failed to get aggregator address and port: {e}")


def start_aggregator(fed_obj):
    """
    Start the aggregator
    Args:
        fed_obj (object): Federation fixture object
    Returns:
        bool: True if successful, else False
    """
    try:
        fed_obj.aggregator.start()
    except Exception as e:
        log.error(f"Failed to start aggregator: {e}")
        raise e
    _check_aggregator_protocol_log(fed_obj.aggregator)
    return True


def ping_from_collaborator(collaborator):
    """
    Ping the aggregator from collaborator to check connectivity
    Args:
        fed_obj (object): Federation fixture object
    Returns:
        bool: True if successful, else False
    """
    log.info(f"Ping the aggregator from {collaborator.name} to check connectivity")
    collaborator.ping_aggregator()
    start_time = time.time()
    time.sleep(5)
    while time.time() - start_time < 30:
        # read the resfile and validate "TLS connection established." message
        with open(collaborator.res_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]
        # print last line
        log.info(f"Last line: {lines[-1]}")
        if any(defaults.COL_TLS_END_MSG in line for line in lines[-7:]):
            log.info(f"Aggregator is reachable from {collaborator.name}")
            return True
        else:
            log.info(f"Aggregator is not reachable from {collaborator.name}. Retrying in 5 seconds...")
            time.sleep(5)
    log.error(f"Aggregator is not reachable from {collaborator.name}")
    return False
