# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import docker
import subprocess
from functools import lru_cache

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.exceptions as ex

log = logging.getLogger(__name__)


def remove_docker_network(list_of_networks=[constants.DOCKER_NETWORK_NAME]):
    """
    Remove docker network.
    Args:
        list_of_networks (list): List of network names to remove.
    """
    client = get_docker_client()
    networks = client.networks.list(names=list_of_networks)
    if not networks:
        log.debug(f"Network(s) {list_of_networks} does not exist")
        return

    for network in networks:
        log.debug(f"Removing network: {network.name}")
        network.remove()
    log.debug(f"Docker network(s) {list_of_networks} removed successfully")


def create_docker_network(list_of_networks=[constants.DOCKER_NETWORK_NAME]):
    """
    Create docker network.
    Args:
        list_of_networks (list): List of network names to create.
    """
    client = get_docker_client()
    networks = client.networks.list(names=list_of_networks)
    if networks:
        log.info(f"Network(s) {list_of_networks} already exists")
        return

    for network_name in list_of_networks:
        log.debug(f"Creating network: {network_name}")
        _ = client.networks.create(network_name)
    log.info(f"Docker network(s) {list_of_networks} created successfully")


def check_docker_image():
    """
    Check if the docker image exists.
    """
    client = get_docker_client()
    images = client.images.list(name=constants.DEFAULT_OPENFL_IMAGE)
    if not images:
        log.error(f"Image {constants.DEFAULT_OPENFL_IMAGE} does not exist")
        raise Exception(f"Image {constants.DEFAULT_OPENFL_IMAGE} does not exist")
    log.debug(f"Image {constants.DEFAULT_OPENFL_IMAGE} exists")


def start_docker_container_with_federation_run(
    participant,
    use_tls=True,
    image=constants.DEFAULT_OPENFL_IMAGE,
    network=constants.DOCKER_NETWORK_NAME,
    env_keyval_list=None,
    security_opt=None,
    mount_mapping=None,
):
    """
    Start the docker container for given participant and sets its container ID.
    IMPORTANT: Internally runs the command to start the federation run.
    Args:
        participant: Participant object (aggregator/collaborator)
        use_tls: Flag to indicate if TLS is enabled. Default is True.
        image: Docker image to use
        network: Docker network to use (default is openfl)
        env_keyval_list: List of environment variables to set.
            Provide in key=val format. For example ["KERAS_HOME=/tmp"]
        security_opt: Security options for the container
        mount_mapping: Mapping of local path to docker path. Format ["local_path:docker_path"]
    Returns:
        container: Docker container object
    """
    try:
        client = get_docker_client()

        # Set Local bind path and Docker container bind path
        if mount_mapping:
            local_participant_path = mount_mapping[0].split(":")[0]
            docker_participant_path = mount_mapping[0].split(":")[1]
        else:
            local_participant_path = participant.workspace_path

            docker_participant_path = f"/{constants.DFLT_DOCKERIZE_IMAGE_NAME}"

        volumes = {
            local_participant_path: {"bind": docker_participant_path, "mode": "rw"},
        }
        log.debug(f"Volumes for {participant.name}: {volumes}")

        environment = {
            "WORKSPACE_PATH": docker_participant_path,
            "NO_PROXY": "aggregator",
            "no_proxy": "aggregator"
        }
        if env_keyval_list:
            for keyval in env_keyval_list:
                key, val = keyval.split("=")
                environment[key] = val

        log.debug(f"Environment variables for {participant.name}: {environment}")

        # Prepare the commands to run based on the participant
        log_file = f"{docker_participant_path}/{participant.name}.log"

        if participant.name == "aggregator":
            start_agg = constants.AGG_START_CMD
            # Handle Fed Eval case
            if participant.eval_scope:
                start_agg += " --task_group evaluation"
            command = ["bash", "-c", f"touch {log_file} && {start_agg} > {log_file} 2>&1"]
        else:
            start_collaborator = f"touch {log_file} && {constants.COL_START_CMD.format(participant.name)} > {log_file} 2>&1"
            if use_tls:
                command = ["bash", "-c", f"{constants.COL_CERTIFY_CMD.format(participant.name)} && {start_collaborator}"]
            else:
                command = ["bash", "-c", start_collaborator]

        log.info(f"Command for {participant.name}: {command}")

        # Start a container from the image
        container = client.containers.run(
            image,
            detach=True,
            user="root",
            auto_remove=False,
            tty=True,
            name=participant.name,
            network=network,
            security_opt=security_opt,
            volumes=volumes,
            environment=environment,
            use_config_proxy=False,  # Do not use proxy for docker container
            command=command
        )
        log.info(f"Container for {participant.name} started with ID: {container.id}")

    except Exception as e:
        raise ex.DockerException(f"Error starting docker container: {e}")

    return container


@lru_cache(maxsize=50)
def get_docker_client():
    """
    Get the Docker client.
    Returns:
        Docker client
    """
    try:
        client = docker.DockerClient(base_url="unix://var/run/docker.sock")
    except Exception as e:
        raise ex.DockerException(f"Error getting docker client: {e}")
    return client


def cleanup_docker_containers(list_of_containers=["aggregator", "collaborator*"]):
    """
    Cleanup the docker containers meant for openfl.
    Args:
        list_of_containers: List of container names to cleanup.
    """
    log.debug("Cleaning up docker containers")

    client = get_docker_client()

    for container_name in list_of_containers:
        containers = client.containers.list(all=True, filters={"name": container_name})
        container_names = []
        # Stop and remove all containers
        for container in containers:
            container.stop()
            container.remove()
            container_names.append(container.name)

        if containers:
            log.info(f"Docker containers {container_names} cleaned up successfully")


def stop_start_docker_participant(participant, action):
    """
    Stop or start the docker participant.
    Args:
        participant: Participant object
        action (str): Action to perform (stop/start)
    """
    if action not in ["stop", "start"]:
        raise ex.DockerException(f"Invalid action {action}")

    client = get_docker_client()

    # List containers with the participant name
    containers = client.containers.list(all=True, filters={"name": participant.name})
    container_names = []

    for container in containers:
        # Restart the participant
        container.stop() if action == "stop" else container.start()
        log.debug(f"Docker {action} successful for {container.name}")
        container_names.append(container.name)

    return True


def build_docker_image(image_name, dockerfile_path):
    """
    Build a docker image.
    Args:
        image_name (str): Name of the image to build
        dockerfile_path (str): Path to the Dockerfile
    """
    log.info(f"Building docker image {image_name}")

    try:
        subprocess.run(
            f"docker build -t {image_name} -f {dockerfile_path} .",
            shell=True,
            check=True,
        )
    except Exception as e:
        raise ex.DockerException(f"Error building docker image: {e}")
