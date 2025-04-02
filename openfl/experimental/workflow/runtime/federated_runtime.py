# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.runtime package FederatedRuntime class."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dill
from tabulate import tabulate

from openfl.experimental.workflow.notebooktools import NotebookTools
from openfl.experimental.workflow.runtime.runtime import Runtime
from openfl.experimental.workflow.transport.grpc.director_client import RuntimeDirectorClient

logger = logging.getLogger(__name__)


class FederatedRuntime(Runtime):
    """FederatedRuntime class, derived from Runtime class.

    Attributes:
        __collaborators (Optional[List[str]]): List of Authorized collaborators
        tls (bool): A flag indicating if TLS should be used for
            connections. Defaults to False.
        director (Optional[Dict[str, Any]]): Dictionary containing director info.
        _runtime_dir_client (RuntimeDirectorClient): The Runtime director client.
        notebook_path (Optional[str]) : Path to the Jupyter notebook
        experiment_submitted (bool): Whether the experiment has been submitted.
        generated_workspace_path (Path): Path to generated workspace
    """

    def __init__(
        self,
        collaborators: Optional[List[str]] = None,
        director: Optional[Dict[str, Any]] = None,
        notebook_path: Optional[str] = None,
        tls: bool = False,
    ) -> None:
        """Initializes the FederatedRuntime object.

        Args:
            collaborators (Optional[List[str]]): List of Authorized collaborators.
                Defaults to None.
            director (Optional[Dict[str, Any]]): Director information. Defaults to None
            notebook_path (Optional[str]): Jupyter notebook path
            tls (bool): Whether to use TLS for the connection.
        """
        super().__init__()
        self.__collaborators = collaborators

        self.tls = tls
        if director:
            self.director = director
            self._fill_certs(
                self.director.get("cert_chain", None),
                self.director.get("api_private_key", None),
                self.director.get("api_cert", None),
            )
            self._runtime_dir_client = self._create_runtime_dir_client()

        self.notebook_path = notebook_path
        self.experiment_submitted = False
        self.generated_workspace_path = Path("./generated_workspace").resolve()

    @staticmethod
    def remove_workspace_archive(archive_path) -> None:
        """
        Removes workspace archive

        Args:
            archive_path (str): Archive file path containing the workspace.
        """
        if os.path.exists(archive_path):
            os.remove(archive_path)

    @property
    def collaborators(self) -> List[str]:
        """Get the names of collaborators.

        Don't give direct access to private attributes.

        Returns:
            List[str]: The names of the collaborators.
        """
        return self.__collaborators

    @collaborators.setter
    def collaborators(self, collaborators: List[str]) -> None:
        """Set the collaborators.

        Args:
            collaborators (List[str]): The list of
                collaborators to set.
        """
        self.__collaborators = collaborators

    def _fill_certs(self, root_certificate, private_key, certificate) -> None:
        """Fill certificates.

        Args:
            root_certificate (Union[Path, str]): The path to the root
                certificate for the TLS connection.
            private_key (Union[Path, str]): The path to the server's private
                key for the TLS connection.
            certificate (Union[Path, str]): The path to the server's
                certificate for the TLS connection.
        """
        if self.tls:
            if not all([root_certificate, private_key, certificate]):
                raise ValueError("Incomplete certificates provided")

            self.root_certificate = Path(root_certificate).absolute()
            self.private_key = Path(private_key).absolute()
            self.certificate = Path(certificate).absolute()
        else:
            self.root_certificate = self.private_key = self.certificate = None

    def _create_runtime_dir_client(self) -> RuntimeDirectorClient:
        """Create a RuntimeDirectorClient instance.

        Returns:
            RuntimeDirectorClient: Instance of the client
        """
        return RuntimeDirectorClient(
            director_host=self.director["director_node_fqdn"],
            director_port=self.director["director_port"],
            tls=self.tls,
            root_certificate=self.root_certificate,
            private_key=self.private_key,
            certificate=self.certificate,
        )

    def prepare_workspace_archive(self) -> Tuple[Path, str]:
        """
        Prepare workspace archive using NotebookTools.

        Returns:
            Tuple[Path, str]: A tuple containing the path of the created
        archive and the experiment name.
        """
        nb_tools = NotebookTools(
            notebook_path=self.notebook_path, output_workspace="./generated_workspace"
        )

        archive_path, exp_name = nb_tools.export(
            director_fqdn=self.director["director_node_fqdn"], tls=self.tls
        )
        return archive_path, exp_name

    def submit_experiment(self, archive_path, exp_name) -> None:
        """
        Submits experiment archive to the director

        Args:
            archive_path (str): Archive file path containing the workspace.
            exp_name (str): The name of the experiment to be submitted.
        """
        try:
            response = self._runtime_dir_client.set_new_experiment(
                archive_path=archive_path, experiment_name=exp_name, col_names=self.__collaborators
            )
            self.experiment_submitted = response.status

            if self.experiment_submitted:
                print(
                    f"\033[92mExperiment {exp_name} was successfully "
                    "submitted to the director!\033[0m"
                )
            else:
                print(f"\033[91mFailed to submit experiment '{exp_name}' to the director.\033[0m")
        finally:
            self.remove_workspace_archive(archive_path)

    def get_flow_state(self) -> Tuple[bool, Any]:
        """
        Retrieve the updated flow status and deserialized flow object.

        Returns:
            status (bool): The flow status.
            flow_object: The deserialized flow object.
        """
        status, flspec_obj = self._runtime_dir_client.get_flow_state()

        # Append generated workspace path to sys.path
        # to allow unpickling of flspec_obj
        sys.path.append(str(self.generated_workspace_path))
        flow_object = dill.loads(flspec_obj)

        return status, flow_object

    def get_envoys(self) -> List[str]:
        """
        Prints the status of Envoys in a formatted way.
        Returns:
            online_envoys (List[str]): List of online envoys.
        """
        # Fetch envoy data
        envoys = self._runtime_dir_client.get_envoys()
        DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
        now = datetime.now().strftime(DATETIME_FORMAT)

        # Prepare the table headers
        headers = ["Name", "Online", "Last Updated", "Experiment Running", "Experiment Name"]
        # Prepare the table rows
        rows = []
        online_envoys = []
        for envoy in envoys.envoy_infos:
            rows.append(
                [
                    envoy.envoy_name,
                    "Yes" if envoy.is_online else "No",
                    datetime.fromtimestamp(envoy.last_updated.seconds).strftime(DATETIME_FORMAT),
                    "Yes" if envoy.is_experiment_running else "No",
                    envoy.experiment_name if envoy.experiment_name else "None",
                ]
            )
            if envoy.is_online:
                online_envoys.append(envoy.envoy_name)

        # Use tabulate to format the table
        result = tabulate(rows, headers=headers, tablefmt="grid")
        # Display the current timestamp
        print(f"Status of Envoys connected to Federation at: {now}\n")
        print(result)
        return online_envoys

    def stream_experiment_stdout(self, experiment_name) -> None:
        """Stream experiment stdout.

        Args:
            experiment_name (str): Name of the experiment.
        """
        if not self.experiment_submitted:
            print("No experiment has been submitted yet.")
            return
        print(f"Getting standard output for experiment: {experiment_name}...")
        for stdout_message_dict in self._runtime_dir_client.stream_experiment_stdout(
            experiment_name
        ):
            print(
                f"Origin: {stdout_message_dict['stdout_origin']}, "
                f"Task: {stdout_message_dict['task_name']}"
                f"\n{stdout_message_dict['stdout_value']}"
            )

    def __repr__(self) -> str:
        """Returns the string representation of the FederatedRuntime object.

        Returns:
            str: The string representation of the FederatedRuntime object.
        """
        return "FederatedRuntime"
