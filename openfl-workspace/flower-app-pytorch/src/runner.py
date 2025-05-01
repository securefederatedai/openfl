from openfl.federated.task.runner import TaskRunner
import subprocess
from logging import getLogger
import time
import os
import numpy as np
from pathlib import Path
import sys
import socket
from src.util import is_safe_path

flwr_home = os.path.join(os.getcwd(), "save/.flwr")
if not is_safe_path(flwr_home):
    raise ValueError("Invalid path for FLWR_HOME")

os.environ["FLWR_HOME"] = flwr_home
os.makedirs(os.environ["FLWR_HOME"], exist_ok=True)

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

        self.sgx_enabled = kwargs.get('sgx_enabled')
        if self.data_loader is None:
            flwr_app_name = kwargs.get('flwr_app_name')
            if self.sgx_enabled:
                install_flower_FAB(flwr_app_name)
            return

        self.model = None
        self.logger = getLogger(__name__)

        self.data_path = self.data_loader.get_node_configs()

        self.client_port = kwargs.get('client_port')
        if self.client_port is None:
            self.client_port = get_dynamic_port()

        self.shutdown_requested = False  # Flag to signal shutdown

    def start_client_adapter(self, interop_server, **kwargs):
        """
        Start the local gRPC server and the Flower SuperNode.

        Args:
            interop_server: The local gRPC server instance.
            **kwargs: Additional parameters, including 'local_server_port'.
        """
        local_server_port = kwargs.get('local_server_port')

        def message_callback():
            self.shutdown_requested = True

        # Set the callback for ending the experiment
        interop_server.set_end_experiment_callback(message_callback)
        interop_server.start_server(local_server_port)

        local_server_port = interop_server.get_port()

        command = [
            "flower-supernode",
            "--insecure",
            "--grpc-adapter",
            "--superlink", f"127.0.0.1:{local_server_port}",
            "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
            "--node-config", f"data-path='{self.data_path}'"
        ]

        if self.sgx_enabled:
            command += ["--isolation", "process"]
            flwr_clientapp_command = [
                "flwr-clientapp",
                "--insecure",
                "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
            ]

        self.logger.info("Starting Flower SuperNode process...")
        supernode_process = subprocess.Popen(command, shell=False)
        interop_server.handle_signals(supernode_process)

        if self.sgx_enabled:
            # Check if port is open before starting the client app
            while not is_port_open('127.0.0.1', local_server_port):
                time.sleep(0.5)

            time.sleep(1) # Add a small delay after confirming the port is open

            self.logger.info("Starting Flower ClientApp process...")
            flwr_clientapp_process = subprocess.Popen(flwr_clientapp_command, shell=False)
            interop_server.handle_signals(flwr_clientapp_process)

        self.logger.info("Press CTRL+C to stop the server and SuperNode process.")

        while not interop_server.termination_event.is_set():
            if self.shutdown_requested:
                if self.sgx_enabled:
                    self.logger.info("Terminating Flower ClientApp process...")
                    interop_server.terminate_supernode_process(flwr_clientapp_process)
                    flwr_clientapp_process.wait()

                self.logger.info("Shutting down the server and SuperNode process...")
                interop_server.terminate_supernode_process(supernode_process)
                interop_server.stop_server()
            time.sleep(0.1)


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

        assert filepath.endswith('.npz'), "Currently, only '.npz' file type is supported."

        # Save the tensor dictionary to a .npz file
        np.savez(filepath, **self.tensor_dict)

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Initialize tensor keys for functions. Currently not implemented."""
        pass


def install_flower_FAB(flwr_app_name):
    """
    Build and install the patch for the Flower application.

    Args:
        flwr_app_name (str): The name of the Flower application to patch.
    """
    flwr_dir = os.environ["FLWR_HOME"]
    os.environ["TMPDIR"] = flwr_dir

    # Run the build command
    subprocess.check_call([
        sys.executable,
        "src/patch/flwr_run_patch.py",
        "build",
        "--app",
        f"./src/{flwr_app_name}"
    ])

    # List .fab files after running the build command
    fab_files = list(Path(flwr_dir).glob("*.fab"))

    # Determine the newest .fab file
    newest_fab_file = max(fab_files, key=os.path.getmtime)

    # Run the install command using the newest .fab file
    subprocess.check_call([
        sys.executable,
        "src/patch/flwr_run_patch.py",
        "install",
        str(newest_fab_file)
    ])

def get_dynamic_port():
    """
    Get a dynamically assigned port number.

    Returns:
        int: An available port number assigned by the operating system.
    """
    # Create a socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to port 0 to let the OS assign an available port
        s.bind(('127.0.0.1', 0))
        # Get the assigned port number
        port = s.getsockname()[1]
    return port

def is_port_open(host, port):
    """Check if a port is open on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0
