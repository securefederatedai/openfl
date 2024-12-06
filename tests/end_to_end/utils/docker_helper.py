# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import docker
import os
from functools import lru_cache

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.ssh_helper as sh

log = logging.getLogger(__name__)


def remove_docker_network():
    """
    Remove docker network.
    """
    client = get_docker_client()
    networks = client.networks.list(names=[constants.DOCKER_NETWORK_NAME])
    if not networks:
        log.info(f"Network {constants.DOCKER_NETWORK_NAME} does not exist")
        return

    for network in networks:
        log.info(f"Removing network: {network.name}")
        network.remove()
    log.info("Docker network removed successfully")


def create_docker_network():
    """
    Create docker network.
    """
    client = get_docker_client()
    networks = client.networks.list(names=[constants.DOCKER_NETWORK_NAME])
    if networks:
        log.info(f"Network {constants.DOCKER_NETWORK_NAME} already exists")
        return

    log.info(f"Creating network: {constants.DOCKER_NETWORK_NAME}")
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
    log.info(f"Image {constants.DEFAULT_OPENFL_IMAGE} exists")


def start_docker_container(
    container_name,
    results_dir,
    workspace_template,
    image=constants.DEFAULT_OPENFL_IMAGE,
):
    """
    Start the docker container with provided name.
    Args:
        container_name: Name of the container
        results_dir: Results directory
        workspace_template: Model name (workspace template)
        image: Docker image to use
    Returns:
        container: Docker container object
    """
    client = get_docker_client()

    working_directory = os.path.join(os.getenv("HOME"), results_dir, workspace_template, container_name, "workspace")

    # Docker container bind path
    bind_path = f"/{results_dir}/{workspace_template}"

    participant_workspace_path = f"{bind_path}/{container_name}/workspace"

    log.info(f"Bind path: {bind_path} and participant workspace path: {participant_workspace_path} and working_directory: {working_directory}")

    # Start a container from the image
    container = client.containers.run(
        image,
        detach=True,
        user='root',
        auto_remove=False,
        tty=True,
        name=container_name,
        network='openfl',
        volumes={working_directory: {'bind': bind_path, 'mode': 'rw'}},
        environment={"WORKSPACE_PATH": participant_workspace_path, "WORKSPACE_TEMPLATE": workspace_template},
        use_config_proxy=False, # Do not use proxy for docker container
    )

    log.info(f"Container for {container_name} started with ID: {container.id}")
    return container


@lru_cache(maxsize=50)
def get_docker_client():
    """
    Get the Docker client.
    Returns:
        Docker client
    """
    client = docker.DockerClient(base_url="unix://var/run/docker.sock")
    return client


def cleanup_docker_containers():
    """
    Cleanup the docker containers meant for openfl.
    """
    log.info("Cleaning up docker containers")

    client = get_docker_client()

    # List all containers related to openfl
    agg_containers = client.containers.list(all=True, filters={'name':'aggregator'})
    col_containers = client.containers.list(all=True, filters={'name':'collaborator*'})

    # itp_tool_container = docker_client.containers.get(itp_tool_container_id)
    containers = agg_containers + col_containers

    container_names = []
    # Stop and remove all containers
    for container in containers:
        container.stop()
        container.remove()
        container_names.append(container.name)

    log.info(f"Docker containers {container_names} cleaned up successfully")
