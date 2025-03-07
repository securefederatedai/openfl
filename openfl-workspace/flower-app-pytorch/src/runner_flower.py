from openfl.federated.task.runner import TaskRunner
import subprocess
from logging import getLogger
import time
import os
import numpy as np
from pathlib import Path
import sys
import socket

os.environ["FLWR_HOME"] = os.path.join(os.getcwd(), "src/.flwr")
os.makedirs(os.environ["FLWR_HOME"], exist_ok=True)

class FlowerTaskRunner(TaskRunner):
    """
    FlowerTaskRunner is a task runner that executes Flower SuperNode
    to initialize the experiment from the client side.

    This class is responsible for starting a local gRPC server and a Flower SuperNode
    in a subprocess. It also provides options for automatic shutdown based on subprocess
    activity.

    Shutdown Options:
    - Manual Shutdown: The server and supernode process can be manually stopped by pressing CTRL+C.
    - Automatic Shutdown: If enabled, the system will monitor the activity of subprocesses and 
      automatically shut down if no new subprocess starts within a certain time frame.
    """
    def __init__(self, **kwargs):
        """
        Initializes the FlowerTaskRunner.

        Args:
            auto_shutdown (bool): Whether to enable automatic shutdown based on subprocess activity.
                Default is True. Set to False for long-lived components.
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__(**kwargs)

        self.patch = kwargs.get('patch')
        if self.data_loader is None:
            flwr_app_name = kwargs.get('flwr_app_name')
            if self.patch:
                install_flower_FAB(flwr_app_name)
            return

        self.model = None
        self.logger = getLogger(__name__)

        self.data_path = self.data_loader.get_node_configs()

        self.client_port = kwargs.get('client_port')
        if self.client_port is None:
            self.client_port = get_dynamic_port()

        self.shutdown_requested = False # Flag signal shutdown

    def start_client_adapter(self, local_grpc_server, **kwargs):
        """
        Starts the local gRPC server and the Flower SuperNode.
        """
        local_server_port = kwargs.get('local_server_port')

        def message_callback():
            self.shutdown_requested = True

        # TODO: Can we isolate the local_grpc_server from the task runner?
        local_grpc_server.set_end_experiment_callback(message_callback)
        local_grpc_server.start_server(local_server_port)

        local_server_port = local_grpc_server.get_port()

        if self.patch:
            command = [
                "python",
                "src/patch/flower_supernode_patch.py",
                "--insecure",
                "--grpc-adapter",
                "--superlink", f"127.0.0.1:{local_server_port}",
                "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
                "--node-config", f"data-path='{self.data_path}'"
            ]
        else:
            command = [
                "flower-supernode",
                "--insecure",
                "--grpc-adapter",
                "--superlink", f"127.0.0.1:{local_server_port}",
                "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
                "--node-config", f"data-path='{self.data_path}'"
            ]

        supernode_process = subprocess.Popen(command, shell=False)
        local_grpc_server.handle_signals(supernode_process)

        self.logger.info("Press CTRL+C to stop the server and SuperNode process.")
        
        try:
            while not local_grpc_server.termination_event.is_set():
                if self.shutdown_requested:
                    local_grpc_server.terminate_supernode_process(supernode_process)
                    local_grpc_server.stop_server()
                time.sleep(0.1)
        except KeyboardInterrupt:
            local_grpc_server.terminate_supernode_process(supernode_process)
            local_grpc_server.stop_server()

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.
        To be framework agnostic, this method will not attempt to load the weights into the model
        and save out the native format. Instead, it will load and save the dictionary directly

        Args:
            tensor_dict (dict): The tensor dictionary.
            with_opt_vars (bool): This argument is inherited from the parent class
                but is not used in the FlowerTaskRunner.
        """
        self.tensor_dict = tensor_dict

    def save_native(
        self,
        filepath,
        **kwargs,
    ):
        """
        Save model weights in a .npz file specified by the filepath.
        The model weights are stored as a dictionary of np.ndarray

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

        # Ensure the file extension is .npz
        assert filepath.endswith('.npz'), "Currently, only '.npz' file type is supported."

        # Save the tensor dictionary to a .npz file
        np.savez(filepath, **self.tensor_dict)

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        pass


def install_flower_FAB(flwr_app_name):
    """Build and install the patch for the Flower application."""
    flwr_dir = os.environ["FLWR_HOME"]
    
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
    # Create a socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to port 0 to let the OS assign an available port
        s.bind(('', 0))
        # Get the assigned port number
        port = s.getsockname()[1]
    return port
    