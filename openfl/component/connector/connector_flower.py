# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import subprocess
import sys
from logging import getLogger

import psutil

from openfl.transport.grpc.interop import FlowerInteropClient
from openfl.utilities.path_check import is_directory_traversal

logger = getLogger(__name__)


class ConnectorFlower:
    """
    A Connector subclass specifically designed for integrating with the Flower framework.
    This class is responsible for constructing and managing the execution of Flower server commands.
    """

    def __init__(
        self,
        superlink_host,
        fleet_api_port,
        exec_api_port,
        serverappio_api_port,
        insecure=True,
        flwr_app_name=None,
        federation_name=None,
        automatic_shutdown=True,
        flwr_dir=None,
        **kwargs,
    ):
        """
        Initialize the ConnectorFlower instance by setting up the necessary server commands.

        Args:
            superlink_host (str): Host address for the Flower SuperLink.
            fleet_api_port (int): Port for the fleet API.
            exec_api_port (int): Port for the exec API.
            serverappio_api_port (int): Port for the serverappio API.
            insecure (bool): Whether to use insecure connections. Defaults to True.
            flwr_app_name (str, optional): Name of the Flower application to run. Defaults to None.
            federation_name (str, optional): Name of the federation. Defaults to None.
            automatic_shutdown (bool, optional): Whether to enable automatic shutdown.
                Defaults to True.
            flwr_dir (str, optional): Directory for Flower app within the OpenFL workspace.
                Plan.yaml configuration defaults to `save/.flwr`
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self._process = None

        self.flwr_dir = flwr_dir
        if is_directory_traversal(self.flwr_dir):
            logger.error("Flower app directory path is out of the OpenFL workspace scope.")
            sys.exit(1)
        else:
            os.makedirs(self.flwr_dir, exist_ok=True)
            os.environ["FLWR_HOME"] = self.flwr_dir

        self.automatic_shutdown = automatic_shutdown
        self.signal_shutdown_sent = False

        self.superlink_params = {
            "insecure": insecure,
            "exec_api_port": exec_api_port,
            "fleet_api_port": fleet_api_port,
            "serverappio_api_port": serverappio_api_port,
        }
        self.superlink_host = superlink_host
        self.flwr_superlink_command = self._build_flwr_superlink_command()

        if flwr_app_name is None or federation_name is None:
            self.flwr_run_params = None
        else:
            self.flwr_run_params = {
                "flwr_app_name": flwr_app_name,
                "federation_name": federation_name,
            }
        self.flwr_run_command = self._build_flwr_run_command() if self.flwr_run_params else None

        self.interop_client = None
        signal.signal(signal.SIGINT, self._handle_sigint)

    def get_interop_client(self):
        """
        Create and return a FlowerInteropClient instance using the superlink parameters.

        Returns:
            FlowerInteropClient: An instance configured with the connector address
            and server rounds.
        """
        connector_port = self.superlink_params.get("fleet_api_port")
        connector_address = f"{self.superlink_host}:{connector_port}"
        self.interop_client = FlowerInteropClient(connector_address, self.automatic_shutdown)
        return self.interop_client

    def _build_flwr_superlink_command(self) -> list[str]:
        """
        Construct the command to initiate the Flower SuperLink based on provided parameters.

        Returns:
            list[str]: A list of command-line arguments for starting the Flower server.
        """

        command = ["flower-superlink", "--fleet-api-type", "grpc-adapter"]

        if self.superlink_params.get("insecure"):
            command += ["--insecure"]

        serverappio_api_port = self.superlink_params.get("serverappio_api_port")
        serverappio_api_address = f"{self.superlink_host}:{serverappio_api_port}"
        command += ["--serverappio-api-address", serverappio_api_address]

        fleet_api_port = self.superlink_params.get("fleet_api_port")
        fleet_api_address = f"{self.superlink_host}:{fleet_api_port}"
        command += ["--fleet-api-address", fleet_api_address]

        exec_api_port = self.superlink_params.get("exec_api_port")
        exec_api_address = f"{self.superlink_host}:{exec_api_port}"
        command += ["--exec-api-address", exec_api_address]

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

        if self.superlink_params["insecure"]:
            command += ["--insecure"]

        serverappio_api_port = self.superlink_params["serverappio_api_port"]
        serverappio_api_address = f"{self.superlink_host}:{serverappio_api_port}"
        command += ["--serverappio-api-address", serverappio_api_address]

        return command

    def is_flwr_serverapp_running(self):
        """
        Determine if the Flower ServerApp subprocess is currently active.

        Returns:
            bool: True if the ServerApp is running, False otherwise.
        """
        if not hasattr(self, "flwr_serverapp_subprocess"):
            logger.debug("[OpenFL Connector] ServerApp was never started.")
            return False

        if self.flwr_serverapp_subprocess.poll() is None:
            logger.debug("[OpenFL Connector] ServerApp is still running.")
            return True

        if not self.signal_shutdown_sent:
            self.signal_shutdown_sent = True
            logger.info(
                "[OpenFL Connector] Experiment has ended. Sending signal "
                "to shut down Flower components."
            )

        return False

    def _stop_flwr_serverapp(self):
        """Terminate the `flwr_serverapp` subprocess if it is still active."""
        if (
            hasattr(self, "flwr_serverapp_subprocess")
            and self.flwr_serverapp_subprocess.poll() is None
        ):
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

        command = ["flwr", "run", f"./src/{flwr_app_name}"]

        if federation_name:
            command.append(federation_name)

        return command

    def start(self):
        """
        Launch the `flower-superlink` and `flwr run` subprocesses
        using the constructed commands.
        """
        if self._process is None:
            logger.info(
                f"[OpenFL Connector] Starting server process: "
                f"{' '.join(self.flwr_superlink_command)}"
            )
            self._process = subprocess.Popen(self.flwr_superlink_command)
            logger.info(f"[OpenFL Connector] Server process started with PID: {self._process.pid}")
        else:
            logger.info("[OpenFL Connector] Server process is already running.")

        if hasattr(self, "flwr_run_command") and self.flwr_run_command:
            logger.info(
                f"[OpenFL Connector] Starting `flwr run` "
                f"subprocess: {' '.join(self.flwr_run_command)}"
            )
            subprocess.run(self.flwr_run_command)

        if hasattr(self, "flwr_serverapp_command") and self.flwr_serverapp_command:
            logger.info(
                f"[OpenFL Connector] Starting server app subprocess: "
                f"{' '.join(self.flwr_serverapp_command)}"
            )
            self.interop_client.set_is_flwr_serverapp_running_callback(
                self.is_flwr_serverapp_running
            )
            self.flwr_serverapp_subprocess = subprocess.Popen(self.flwr_serverapp_command)

    def stop(self):
        """Terminate the `flower-superlink` subprocess and any associated processes."""
        self._stop_flwr_serverapp()
        if self._process:
            try:
                logger.info(
                    f"[OpenFL Connector] Stopping server process with PID: {self._process.pid}..."
                )
                main_process = psutil.Process(self._process.pid)
                sub_processes = main_process.children(recursive=True)
                for sub_process in sub_processes:
                    logger.info(
                        (
                            f"[OpenFL Connector] Stopping server subprocess "
                            f"with PID: {sub_process.pid}..."
                        )
                    )
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
