# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import docker
import os
from functools import lru_cache

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.exceptions as ex

log = logging.getLogger(__name__)


def remove_docker_network():
    """
    Remove docker network.
    """
    client = get_docker_client()
    networks = client.networks.list(names=[constants.DOCKER_NETWORK_NAME])
    if not networks:
        log.debug(f"Network {constants.DOCKER_NETWORK_NAME} does not exist")
        return

    for network in networks:
        log.debug(f"Removing network: {network.name}")
        network.remove()
    log.debug("Docker network removed successfully")


def create_docker_network():
    """
    Create docker network.
    """
    client = get_docker_client()
    networks = client.networks.list(names=[constants.DOCKER_NETWORK_NAME])
    if networks:
        log.info(f"Network {constants.DOCKER_NETWORK_NAME} already exists")
        return

    log.debug(f"Creating network: {constants.DOCKER_NETWORK_NAME}")
    network = client.networks.create(constants.DOCKER_NETWORK_NAME)
    log.info(f"Network {network.name} created successfully")


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


def start_docker_container(
    container_name,
    workspace_path,
    local_bind_path,
    image=constants.DEFAULT_OPENFL_IMAGE,
    network=constants.DOCKER_NETWORK_NAME,
    env_keyval_list=None,
    security_opt=None,
    mount_mapping=None,
):
    """
    Start the docker container with provided name.
    Args:
        container_name: Name of the container
        workspace_path: Workspace path
        local_bind_path: Local bind path
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
            local_participant_path = os.path.join(local_bind_path, container_name, "workspace")
            docker_participant_path = f"{workspace_path}/{container_name}/workspace"

        volumes = {
            local_participant_path: {"bind": docker_participant_path, "mode": "rw"},
        }
        log.debug(f"Volumes for {container_name}: {volumes}")

        environment = {
            "WORKSPACE_PATH": docker_participant_path,
            "NO_PROXY": "aggregator",
            "no_proxy": "aggregator"
        }
        if env_keyval_list:
            for keyval in env_keyval_list:
                key, val = keyval.split("=")
                environment[key] = val

        log.debug(f"Environment variables for {container_name}: {environment}")
        # Start a container from the image
        container = client.containers.run(
            image,
            detach=True,
            user="root",
            auto_remove=False,
            tty=True,
            name=container_name,
            network=network,
            security_opt=security_opt,
            volumes=volumes,
            environment=environment,
            use_config_proxy=False,  # Do not use proxy for docker container
        )
        log.info(f"Container for {container_name} started with ID: {container.id}")

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


def cleanup_docker_containers():
    """
    Cleanup the docker containers meant for openfl.
    """
    log.debug("Cleaning up docker containers")

    client = get_docker_client()

    # List all containers related to openfl
    agg_containers = client.containers.list(all=True, filters={"name": "aggregator"})
    col_containers = client.containers.list(all=True, filters={"name": "collaborator*"})
    containers = agg_containers + col_containers
    container_names = []
    # Stop and remove all containers
    for container in containers:
        container.stop()
        container.remove()
        container_names.append(container.name)

    if containers:
        log.info(f"Docker containers {container_names} cleaned up successfully")
