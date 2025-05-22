# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from openfl.federated.task.runner import TaskRunner
from openfl.utilities.path_check import is_directory_traversal
from openfl.utilities.utils import generate_port

logger = logging.getLogger(__name__)


class FlowerTaskRunner(TaskRunner):
    """
    FlowerTaskRunner is a task runner that executes the Flower SuperNode
    to initialize and manage experiments from the client side.

    This class is responsible for starting a local gRPC server and a Flower SuperNode
    in a subprocess. It provides options for both manual and automatic shutdown based on
    subprocess activity.
    """

    def __init__(self, **kwargs):
        """
        Initialize the FlowerTaskRunner.

        Args:
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__(**kwargs)

        self.flwr_dir = kwargs.get("flwr_dir")
        if is_directory_traversal(self.flwr_dir):
            logger.error("Flower app directory path is out of the OpenFL workspace scope.")
            sys.exit(1)
        else:
            os.makedirs(self.flwr_dir, exist_ok=True)
            os.environ["FLWR_HOME"] = self.flwr_dir

        if self.data_loader is None:
            flwr_app_name = kwargs.get("flwr_app_name")
            install_flower_FAB(flwr_app_name)
            return

        self.sgx_enabled = kwargs.get("sgx_enabled")

        self.model = None

        self.data_path = self.data_loader.get_node_configs()

        self.shutdown_requested = False  # Flag to signal shutdown

    def start_client_adapter(self, col_name=None, round_num=None, input_tensor_dict=None, **kwargs):
        """
        Start the FlowerInteropServer and the Flower SuperNode.

        Args:
            col_name (str, optional): The collaborator name. Defaults to None.
            round_num (int, optional): The current round number. Defaults to None.
            input_tensor_dict (dict, optional): The input tensor dictionary. Defaults to None.
            **kwargs: Additional parameters for configuration.
                includes:
                    interop_server (object): The FlowerInteropServer instance.
                    interop_server_host (str): The address of the interop server.
                    clientappio_api_port (int): The port for the clientappio API.
                    local_simulation (bool): Flag for local simulation to dynamically adjust ports.
                    interop_server_port (int): The port for the interop server.
        """

        def message_callback():
            self.shutdown_requested = True

        interop_server = kwargs.get("interop_server")
        interop_server_host = kwargs.get("interop_server_host")
        interop_server_port = kwargs.get("interop_server_port")
        clientappio_api_port = kwargs.get("clientappio_api_port")

        if kwargs.get("local_simulation"):
            # Dynamically adjust ports for local simulation
            logger.info(f"Adjusting ports for local simulation: {col_name}")

            interop_server_port = get_dynamic_port(interop_server_port, col_name)
            clientappio_api_port = get_dynamic_port(clientappio_api_port, col_name)

            logger.info(f"Adjusted interop_server_port: {interop_server_port}")
            logger.info(f"Adjusted clientappio_api_port: {clientappio_api_port}")

        interop_server.set_end_experiment_callback(message_callback)
        interop_server.start_server(interop_server_host, interop_server_port)

        command = [
            "flower-supernode",
            "--insecure",
            "--grpc-adapter",
            "--superlink",
            f"{interop_server_host}:{interop_server_port}",
            "--clientappio-api-address",
            f"{interop_server_host}:{clientappio_api_port}",
            "--node-config",
            f"data-path='{self.data_path}'",
        ]

        if self.sgx_enabled:
            command += ["--isolation", "process"]
            flwr_clientapp_command = [
                "flwr-clientapp",
                "--insecure",
                "--clientappio-api-address",
                f"{interop_server_host}:{clientappio_api_port}",
            ]

        logger.info("Starting Flower SuperNode process...")
        supernode_process = subprocess.Popen(command, shell=False)
        interop_server.handle_signals(supernode_process)

        if self.sgx_enabled:
            # Check if port is open before starting the client app
            while not is_port_open(interop_server_host, interop_server_port):
                time.sleep(0.5)

            time.sleep(1)  # Add a small delay after confirming the port is open

            logger.info("Starting Flower ClientApp process...")
            flwr_clientapp_process = subprocess.Popen(flwr_clientapp_command, shell=False)
            interop_server.handle_signals(flwr_clientapp_process)

        logger.info("Press CTRL+C to stop the server and SuperNode process.")

        while not interop_server.termination_event.is_set():
            if self.shutdown_requested:
                if self.sgx_enabled:
                    logger.info("Terminating Flower ClientApp process...")
                    interop_server.terminate_supernode_process(flwr_clientapp_process)
                    flwr_clientapp_process.wait()

                logger.info("Shutting down the server and SuperNode process...")
                interop_server.terminate_supernode_process(supernode_process)
                interop_server.stop_server()
            time.sleep(0.1)

        # Collaborator expects these dictionaries, but they are not used in this context
        # as Flower will handle the tensors internally.
        global_output_tensor_dict = {}
        local_output_tensor_dict = {}

        return global_output_tensor_dict, local_output_tensor_dict

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """
        Set the tensor dictionary for the task runner.

        This method is framework agnostic and does not attempt to load the weights into the model
        or save out the native format. Instead, it directly loads and saves the dictionary.

        Args:
            tensor_dict (dict): The tensor dictionary.
            with_opt_vars (bool): This argument is inherited from the parent class
                but is not used in the FlowerTaskRunner.
        """
        self.tensor_dict = tensor_dict

    def save_native(self, filepath, **kwargs):
        """
        Save model weights to a .npz file specified by the filepath.

        The model weights are stored as a dictionary of np.ndarray.

        Args:
            filepath (str): Path to the .npz file to be created by np.savez().
            **kwargs: Additional parameters (currently not used).

        Returns:
            None

        Raises:
            AssertionError: If the file extension is not '.npz'.
        """
        # Ensure the file extension is .npz
        if isinstance(filepath, Path):
            filepath = str(filepath)

        assert filepath.endswith(".npz"), "Currently, only '.npz' file type is supported."

        # Save the tensor dictionary to a .npz file
        np.savez(filepath, **self.tensor_dict)

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Initialize tensor keys for functions. Currently not implemented."""
        pass

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """Get tensor keys for functions. Return empty dict."""
        return {}


def install_flower_FAB(flwr_app_name):
    """
    Build and install Flower application.

    Args:
        flwr_app_name (str): The name of the Flower application.
    """
    # Run the build command
    build_command = ["flwr", "build", "--app", os.path.join("src", flwr_app_name)]
    subprocess.check_call(build_command)

    # List .fab files after running the build command
    fab_files = list(Path.cwd().glob("*.fab"))

    # Determine the newest .fab file
    newest_fab_file = max(fab_files, key=os.path.getmtime)

    # Run the install command using the newest .fab file
    install_command = ["flwr", "install", str(newest_fab_file)]
    subprocess.check_call(install_command)
    os.remove(newest_fab_file)


def get_dynamic_port(base_port, collaborator_name):
    """
    Get a dynamically assigned port number based on collaborator name and base port.
    This is only necessary for local simulation in order to avoid port conflicts.

    Returns:
        int: The dynamically assigned port number.
    """
    combined_string = f"{base_port}--{collaborator_name}"
    hash_object = hashlib.md5(combined_string.encode())
    hash_value = hash_object.hexdigest()
    return generate_port(hash_value)


def is_port_open(host, port):
    """Check if a port is open on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0
