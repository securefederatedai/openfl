# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Notebook Tools module."""

import shutil
from importlib import import_module
from logging import getLogger
from pathlib import Path
from shutil import copytree
from typing import Tuple

import yaml

from openfl.experimental.workflow.federated.plan import Plan
from openfl.experimental.workflow.interface.cli.cli_helper import print_tree
from openfl.experimental.workflow.notebooktools.code_analyzer import CodeAnalyzer

logger = getLogger(__name__)


class NotebookTools:
    """The class is responsible for converting workflow API
    into an OpenFL workspace

    Attributes:
        notebook_path: Absolute path of jupyter notebook.
        template_workspace_path: Path to template workspace provided with
            OpenFL.
        output_workspace_path: Output directory for new generated workspace
            (default="/tmp").
    """

    def __init__(self, notebook_path: str, output_workspace: str) -> None:
        """Initialize a NotebookTools object.
        Args:
            notebook_path (str): The path to the Jupyter notebook that needs to be converted.
            output_workspace (str): The directory where the converted workspace will be saved
                workspace
        """
        self.notebook_path = Path(notebook_path).resolve()
        # Check if the Jupyter notebook exists
        if not self.notebook_path.exists() or not self.notebook_path.is_file():
            raise FileNotFoundError(f"The Jupyter notebook at {notebook_path} does not exist.")

        self.output_workspace_path = Path(output_workspace).resolve()
        # Regenerate the workspace if it already exists
        if self.output_workspace_path.exists():
            shutil.rmtree(self.output_workspace_path)
        self.output_workspace_path.parent.mkdir(parents=True, exist_ok=True)

        self.template_workspace_path = (
            Path(f"{__file__}")
            .parent.parent.parent.parent.parent.joinpath(
                "openfl-workspace",
                "experimental",
                "workflow",
                "AggregatorBasedWorkflow",
                "template_workspace",
            )
            .resolve(strict=True)
        )

        # Copy template workspace to output directory
        copytree(self.template_workspace_path, self.output_workspace_path)

        logger.info(f"Copied template workspace to {self.output_workspace_path}")

        # Initialize CodeAnalyzer object
        self.code_analyzer = CodeAnalyzer()
        # Initialize the script with in the CodeAnalyzer
        self.code_analyzer._initialize_script(self.notebook_path, self.output_workspace_path)

    @classmethod
    def export_federated(
        cls, notebook_path: str, output_workspace: str, director_fqdn: str, tls: bool = False
    ) -> Tuple[str, str]:
        """Exports workspace for FederatedRuntime.

        Args:
            notebook_path (str): Path to the Jupyter notebook.
            output_workspace (str): Path for the generated workspace directory.
            director_fqdn (str): Fully qualified domain name of the director node.
            tls (bool, optional): Whether to use TLS for the connection.

        Returns:
            Tuple[str, str]: A tuple containing:
                (archive_path, flow_class_name).
        """
        instance = cls(notebook_path, output_workspace)
        instance.generate_requirements()
        instance.generate_plan_yaml(director_fqdn, tls)
        instance._clean_generated_workspace()
        print_tree(output_workspace, level=2)

    @classmethod
    def export(cls, notebook_path: str, output_workspace: str) -> None:
        """Exports workspace to output_workspace.
        Args:
            notebook_path (str): Path to the Jupyter notebook.
            output_workspace (str): Path for the generated workspace directory.
        """
        instance = cls(notebook_path, output_workspace)
        instance.generate_requirements()
        instance.generate_plan_yaml()
        instance.generate_data_yaml()
        print_tree(output_workspace, level=2)

    def generate_requirements(self) -> None:
        """Extracts pip libraries mentioned in exported python script and append
        in workspace/requirements.txt.
        """
        requirements, line_numbers, data = self.code_analyzer.get_requirements()

        requirements_filepath = str(
            self.output_workspace_path.joinpath("requirements.txt").resolve()
        )

        # Write libraries found in requirements.txt
        with open(requirements_filepath, "a") as f:
            f.writelines(requirements)

        # Delete pip requirements from the python script to ensure it can be imported
        self.code_analyzer.remove_lines(data, line_numbers)

    def _clean_generated_workspace(self) -> None:
        """
        Remove cols.yaml and data.yaml from the generated workspace
        as these are not needed in FederatedRuntime (Director based workflow)

        """
        cols_file = self.output_workspace_path.joinpath("plan", "cols.yaml")
        data_file = self.output_workspace_path.joinpath("plan", "data.yaml")

        if cols_file.exists():
            cols_file.unlink()
        if data_file.exists():
            data_file.unlink()

    def __read_yaml(self, path) -> dict:
        """Reads a YAML file and returns its contents.
        Args:
            path (str): The path to the YAML file.

        Returns:
            dict: The contents of the YAML file.
        """
        with open(path, "r") as y:
            return yaml.safe_load(y)

    def __write_yaml(self, path, data) -> None:
        """Writes data to a YAML file.
        Args:
            path (str): The path to the YAML file.
            data (dict): The data to write to the YAML file.
        """
        with open(path, "w") as y:
            yaml.safe_dump(data, y)

    def generate_plan_yaml(self, director_fqdn: str = None, tls: bool = False) -> None:
        """Generate the plan.yaml file containing the federated learning flow configuration
        Args:
            director_fqdn (str): Fully qualified domain name of the director node.
            tls (bool, optional): Whether to use TLS for the connection.
        """
        flspec = import_module("openfl.experimental.workflow.interface").FLSpec
        # Get the flow_class details
        flow_details = self.code_analyzer.get_flow_class_details(flspec)
        # Analyze and generate plan configuration
        flow_config = self.code_analyzer.analyze_flow_configuration(flow_details)

        # Determine the path for the plan.yaml file
        plan = self.output_workspace_path.joinpath("plan", "plan.yaml").resolve()

        ## Read or initialize the YAML data
        data = self._read_or_initialize_plan_yaml(plan)

        # Update the plan_configuration with the analyzed flow configuration
        data["federated_flow"].update(flow_config["federated_flow"])

        # Updating the aggregator address with director's hostname and tls settings in plan.yaml
        if director_fqdn:
            network_settings = Plan.parse(plan).config["network"]
            data["network"] = network_settings
            data["network"]["settings"]["agg_addr"] = director_fqdn
            data["network"]["settings"]["tls"] = tls

        # Write the updated plan configuraiton to the plan.yaml file
        self.__write_yaml(plan, data)

    def generate_data_yaml(self) -> None:
        """Generate data.yaml with runtime configuration"""
        # Ensure flow_class is available
        flow_class_name = self._ensure_flow_class()

        # Get runtime information using CodeAnalyzer
        runtime, flow_name = self.code_analyzer.get_runtime_info(flow_class_name)

        # Determine the path for the data.yaml
        data_yaml = self.output_workspace_path.joinpath("plan", "data.yaml").resolve()

        # Read or initialize the YAML data
        data = self._read_or_initialize_data_yaml(data_yaml)

        # Initiaize runtime name
        runtime_name = "local_runtime"

        # Process aggregator information using CodeAnalyzer
        runtime_created = self.code_analyzer.process_aggregator(
            runtime, data, flow_name, runtime_name
        )

        # Process collaborator information using CodeAnalyzer
        data = self.code_analyzer.process_collaborators(
            runtime, data, flow_name, runtime_created, runtime_name
        )

        # Write updated data configuration to the data.yaml file
        self.__write_yaml(data_yaml, data)

    def _ensure_flow_class(self) -> str:
        """Ensure flow class is available and returns its name"""
        if not hasattr(self, "flow_class_name"):
            flspsec = import_module("openfl.experimental.workflow.interface").FLSpec
            flow_details = self.code_analyzer.get_flow_class_details(flspsec)
            self.flow_class_name = flow_details["flow_class_name"]

        return self.flow_class_name

    def _read_or_initialize_plan_yaml(self, plan_yaml) -> dict:
        """Read or initialize the plan YAML data.
        Args:
            plan_yaml (Path): The path to the plan.yaml file.

        Returns:
            dict: The data dictionary from plan.yaml.
        """
        data = self.__read_yaml(plan_yaml)
        if data is None:
            data = {}
            data["federated_flow"] = {"settings": {}, "template": ""}
        return data

    def _read_or_initialize_data_yaml(self, data_yaml) -> dict:
        """Read or initialize the YAML data.
        Args:
            data_yaml (Path): The path to the data.yaml file.

        Returns:
            dict: The data dictionary from data.yaml
        """
        data = self.__read_yaml(data_yaml)
        return data if data is not None else {}
