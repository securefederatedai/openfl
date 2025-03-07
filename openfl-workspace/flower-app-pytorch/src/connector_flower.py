import subprocess
from src.connector import Connector
from src.grpc.connector.flower.local_grpc_client import LocalGRPCClient

import subprocess
import psutil

import os
os.environ["FLWR_HOME"] = os.path.join(os.getcwd(), "src/.flwr")
os.makedirs(os.environ["FLWR_HOME"], exist_ok=True)

class ConnectorFlower(Connector):
    """
    Connector subclass for the Flower framework.
    Responsible for generating the Flower server command.
    """

    def __init__(self, 
                 superlink_params: dict, 
                 flwr_run_params: dict = None, 
                 automatic_shutdown: bool = True, 
                 **kwargs):
        """
        Initialize ConnectorFlower by building the server command.
        
        Args:
            superlink_params (dict): A dictionary of Flower server settings.
            flwr_run_params (dict): A dictionary containing the Flower run parameters.
        """
        super().__init__(component_name="Flower")
        self._process = None
        
        self.automatic_shutdown = automatic_shutdown
        self.signal_shutdown_sent = False

        self.superlink_params = superlink_params
        self.flwr_superlink_command = self._build_flwr_superlink_command()

        self.flwr_run_params = flwr_run_params
        self.flwr_run_command = self._build_flwr_run_command() if self.flwr_run_params else None

        self.local_grpc_client = self._get_local_grpc_client()

    def _get_local_grpc_client(self):
        """
        Create and return a LocalGRPCClient instance based on superlink_params
        and the number of server rounds from the pyproject.toml file.

        Returns:
            LocalGRPCClient: An instance of LocalGRPCClient initialized with the
                             connector address and number of server rounds.
        """
        connector_address = self.superlink_params.get("fleet-api-address", "0.0.0.0:9092")
        return LocalGRPCClient(connector_address, self.automatic_shutdown)

    def _build_flwr_superlink_command(self) -> list[str]:
        """
        Build the command to start the Flower SuperLink based on superlink_params.

        Returns:
            list[str]: A list representing the Flower server start command.
        """
        if self.superlink_params.get("patch"):
            command = ["python", "src/patch/flower_superlink_patch.py", "--fleet-api-type", "grpc-adapter"]
        else:
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
        Build the command to start the Flower ServerApp based on superlink_params.

        Returns:
            list[str]: A list representing the Flower server start command.
        """
        command = ["flwr-serverapp", "--run-once"]

        if "insecure" in self.superlink_params and self.superlink_params["insecure"]:
            command += ["--insecure"]

        if "serverappio-api-address" in self.superlink_params:
            command += ["--serverappio-api-address", str(self.superlink_params["serverappio-api-address"])]

        return command

    def is_flwr_serverapp_running(self):
        """
        Check if the flwr_serverapp subprocess is still running.

        Returns:
            bool: True if the ServerApp is running, False otherwise.
        """
        if not hasattr(self, 'flwr_serverapp_subprocess'):
            self.logger.debug("[OpenFL Connector] ServerApp was never started.")
            return False

        if self.flwr_serverapp_subprocess.poll() is None:
            self.logger.debug("[OpenFL Connector] ServerApp is still running.")
            return True

        if not self.signal_shutdown_sent:
            self.signal_shutdown_sent = True
            self.logger.info("[OpenFL Connector] Experiment has ended. Sending signal to shut down Flower components.")

        return False
    
    def _stop_flwr_serverapp(self):
        """Stop the `flwr_serverapp` subprocess if it is still running."""
        if hasattr(self, 'flwr_serverapp_subprocess') and self.flwr_serverapp_subprocess.poll() is None:
            self.logger.debug("[OpenFL Connector] ServerApp still running. Stopping...")
            self.flwr_serverapp_subprocess.terminate()
            try:
                self.flwr_serverapp_subprocess.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.flwr_serverapp_subprocess.kill()

    def _build_flwr_run_command(self) -> list[str]:
        """
        Build the `flwr run` command to run the Flower application.
        
        Returns:
            list[str]: A list representing the flwr_run command.
        """
        federation_name = self.flwr_run_params.get("federation_name")
        flwr_app_name = self.flwr_run_params.get("flwr_app_name")

        if self.flwr_run_params.get("patch"):
            command = ["python", "src/patch/flwr_run_patch.py", "run", f"./src/{flwr_app_name}"]
        else:
            command = ["flwr", "run", f"./src/{flwr_app_name}"]

        if federation_name:
            command.append(federation_name)

        return command

    def start(self):
        """Start the `flower-superlink` and `flwr run` subprocesses with the provided commands."""
        if self._process is None:
            self.logger.info(f"[OpenFL Connector] Starting server process: {' '.join(self.flwr_superlink_command)}")
            self._process = subprocess.Popen(self.flwr_superlink_command)
            self.logger.info(f"[OpenFL Connector] Server process started with PID: {self._process.pid}")
        else:
            self.logger.info("[OpenFL Connector] Server process is already running.")
        
        if hasattr(self, 'flwr_run_command') and self.flwr_run_command:
            self.logger.info(f"[OpenFL Connector] Starting `flwr run` subprocess: {' '.join(self.flwr_run_command)}")
            subprocess.run(self.flwr_run_command)

        if hasattr(self, 'flwr_serverapp_command') and self.flwr_serverapp_command:
            self.local_grpc_client.set_is_flwr_serverapp_running_callback(self.is_flwr_serverapp_running)
            self.flwr_serverapp_subprocess = subprocess.Popen(self.flwr_serverapp_command)

    def stop(self):
        """Stop the `flower-superlink` subprocess."""
        self._stop_flwr_serverapp()
        if self._process:
            try:
                self.logger.info(f"[OpenFL Connector] Stopping server process with PID: {self._process.pid}...")
                main_process = psutil.Process(self._process.pid)
                sub_processes = main_process.children(recursive=True)
                for sub_process in sub_processes:
                    self.logger.info(f"[OpenFL Connector] Stopping server subprocess with PID: {sub_process.pid}...")
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
                self.logger.info("[OpenFL Connector] Server process stopped.")
            except Exception as e:
                self.logger.debug(f"[OpenFL Connector] Error during graceful shutdown: {e}")
                self._process.kill()
                self.logger.info("[OpenFL Connector] Server process forcefully terminated.")
        else:
            self.logger.info("[OpenFL Connector] No server process is currently running.")