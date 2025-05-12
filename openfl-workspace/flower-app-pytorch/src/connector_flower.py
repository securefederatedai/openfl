from logging import getLogger
logger = getLogger(__name__)

import psutil
import subprocess
import sys
import signal

from src.grpc.connector.flower.interop_client import FlowerInteropClient
from src.util import is_safe_path

import os

flwr_home = os.path.join(os.getcwd(), "save/.flwr")
if not is_safe_path(flwr_home):
    raise ValueError("Invalid path for FLWR_HOME")

os.environ["FLWR_HOME"] = flwr_home
os.makedirs(os.environ["FLWR_HOME"], exist_ok=True)

class ConnectorFlower:
    """
    A Connector subclass specifically designed for integrating with the Flower framework.
    This class is responsible for constructing and managing the execution of Flower server commands.
    """

    def __init__(self,
                 superlink_params: dict,
                 flwr_run_params: dict = None,
                 automatic_shutdown: bool = True,
                 **kwargs):
        """
        Initialize the ConnectorFlower instance by setting up the necessary server commands.

        Args:
            superlink_params (dict): Configuration settings for the Flower server.
            flwr_run_params (dict, optional): Parameters for running the Flower application.
            automatic_shutdown (bool, optional): Flag to enable automatic shutdown of the server. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self._process = None

        self.automatic_shutdown = automatic_shutdown
        self.signal_shutdown_sent = False

        self.superlink_params = superlink_params
        self.flwr_superlink_command = self._build_flwr_superlink_command()

        self.flwr_run_params = flwr_run_params
        self.flwr_run_command = self._build_flwr_run_command() if self.flwr_run_params else None

        self.interop_client = None
        signal.signal(signal.SIGINT, self._handle_sigint)

    def get_interop_client(self):
        """
        Create and return a LocalGRPCClient instance using the superlink parameters.

        Returns:
            LocalGRPCClient: An instance configured with the connector address and server rounds.
        """
        connector_address = self.superlink_params.get("fleet-api-address", "0.0.0.0:9092")
        self.interop_client = FlowerInteropClient(connector_address, self.automatic_shutdown)
        return self.interop_client

    def _build_flwr_superlink_command(self) -> list[str]:
        """
        Construct the command to initiate the Flower SuperLink based on provided parameters.

        Returns:
            list[str]: A list of command-line arguments for starting the Flower server.
        """

        command = ["flower-superlink", "--fleet-api-type", "grpc-adapter"]

        if "insecure" in self.superlink_params and self.superlink_params["insecure"]:
            command += ["--insecure"]

        if "serverappio-api-address" in self.superlink_params:
            command += ["--serverappio-api-address", str(self.superlink_params["serverappio-api-address"])]
            # flwr default: 0.0.0.0:9091

        if "fleet-api-address" in self.superlink_params:
            command += ["--fleet-api-address", str(self.superlink_params["fleet-api-address"])]
            # flwr default: 0.0.0.0:9092

        if "exec-api-address" in self.superlink_params:
            command += ["--exec-api-address", str(self.superlink_params["exec-api-address"])]
            # flwr default: 0.0.0.0:9093

        if self.automatic_shutdown:
            command += ["--isolation", "process"]
            self.flwr_serverapp_command = self._build_flwr_serverapp_command()
            # flwr will default to "--isolation subprocess"

        return command

    def _build_flwr_serverapp_command(self) -> list[str]:
        """
        Construct the command to start the Flower ServerApp based on superlink parameters.

        Returns:
            list[str]: A list of command-line arguments for starting the Flower ServerApp.
        """
        command = ["flwr-serverapp", "--run-once"]

        if "insecure" in self.superlink_params and self.superlink_params["insecure"]:
            command += ["--insecure"]

        if "serverappio-api-address" in self.superlink_params:
            command += ["--serverappio-api-address", str(self.superlink_params["serverappio-api-address"])]

        return command

    def is_flwr_serverapp_running(self):
        """
        Determine if the Flower ServerApp subprocess is currently active.

        Returns:
            bool: True if the ServerApp is running, False otherwise.
        """
        if not hasattr(self, 'flwr_serverapp_subprocess'):
            logger.debug("[OpenFL Connector] ServerApp was never started.")
            return False

        if self.flwr_serverapp_subprocess.poll() is None:
            logger.debug("[OpenFL Connector] ServerApp is still running.")
            return True

        if not self.signal_shutdown_sent:
            self.signal_shutdown_sent = True
            logger.info("[OpenFL Connector] Experiment has ended. Sending signal to shut down Flower components.")

        return False

    def _stop_flwr_serverapp(self):
        """Terminate the `flwr_serverapp` subprocess if it is still active."""
        if hasattr(self, 'flwr_serverapp_subprocess') and self.flwr_serverapp_subprocess.poll() is None:
            logger.debug("[OpenFL Connector] ServerApp still running. Stopping...")
            self.flwr_serverapp_subprocess.terminate()
            try:
                self.flwr_serverapp_subprocess.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.flwr_serverapp_subprocess.kill()

    def _build_flwr_run_command(self) -> list[str]:
        """
        Construct the `flwr run` command to execute the Flower application.

        Returns:
            list[str]: A list of command-line arguments for running the Flower application.
        """
        federation_name = self.flwr_run_params.get("federation_name")
        flwr_app_name = self.flwr_run_params.get("flwr_app_name")

        os.environ["TMPDIR"] = os.environ["FLWR_HOME"]

        if self.flwr_run_params.get("sgx_enabled"):
            command = ["python", "src/patch/flwr_run_patch.py", "run", f"./src/{flwr_app_name}"]
        else:
            command = ["flwr", "run", f"./src/{flwr_app_name}"]

        if federation_name:
            command.append(federation_name)

        return command

    def start(self):
        """Launch the `flower-superlink` and `flwr run` subprocesses using the constructed commands."""
        if self._process is None:
            logger.info(f"[OpenFL Connector] Starting server process: {' '.join(self.flwr_superlink_command)}")
            self._process = subprocess.Popen(self.flwr_superlink_command)
            logger.info(f"[OpenFL Connector] Server process started with PID: {self._process.pid}")
        else:
            logger.info("[OpenFL Connector] Server process is already running.")

        if hasattr(self, 'flwr_run_command') and self.flwr_run_command:
            logger.info(f"[OpenFL Connector] Starting `flwr run` subprocess: {' '.join(self.flwr_run_command)}")
            subprocess.run(self.flwr_run_command)

        if hasattr(self, 'flwr_serverapp_command') and self.flwr_serverapp_command:
            self.interop_client.set_is_flwr_serverapp_running_callback(self.is_flwr_serverapp_running)
            self.flwr_serverapp_subprocess = subprocess.Popen(self.flwr_serverapp_command)

    def stop(self):
        """Terminate the `flower-superlink` subprocess and any associated processes."""
        self._stop_flwr_serverapp()
        if self._process:
            try:
                logger.info(f"[OpenFL Connector] Stopping server process with PID: {self._process.pid}...")
                main_process = psutil.Process(self._process.pid)
                sub_processes = main_process.children(recursive=True)
                for sub_process in sub_processes:
                    logger.info(f"[OpenFL Connector] Stopping server subprocess with PID: {sub_process.pid}...")
                    sub_process.terminate()
                _, still_alive = psutil.wait_procs(sub_processes, timeout=1)
                for p in still_alive:
                    p.kill()
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None
                logger.info("[OpenFL Connector] Server process stopped.")
            except Exception as e:
                logger.debug(f"[OpenFL Connector] Error during graceful shutdown: {e}")
                self._process.kill()
                logger.info("[OpenFL Connector] Server process forcefully terminated.")
        else:
            logger.info("[OpenFL Connector] No server process is currently running.")

    def _handle_sigint(self, signum, frame):
        """Handle the SIGINT signal (Ctrl+C) to cleanly stop the server process and its children."""
        logger.info("[OpenFL Connector] SIGINT received. Terminating server process...")
        self.stop()
        sys.exit(0)
