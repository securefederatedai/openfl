# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Notebook Tools module."""

import shutil
from importlib import import_module
from logging import getLogger
from pathlib import Path
from shutil import copytree
from typing import Tuple

from openfl.experimental.workflow.federated.plan import Plan
from openfl.experimental.workflow.interface.cli.cli_helper import print_tree
from openfl.experimental.workflow.notebooktools.code_analyzer import CodeAnalyzer

logger = getLogger(__name__)


class NotebookTools:
    """Class to convert LocalRuntime Jupyter notebook based on Workflow API into a
    workspace that could be deployed on distributed infrastructure

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
        self.code_analyzer = CodeAnalyzer(self.notebook_path, self.output_workspace_path)

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
        return instance.generate_experiment_archive()

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

    def generate_experiment_archive(self) -> Tuple[str, str]:
        """
        Create archive of the generated workspace

        Returns:
            Tuple[str, str]: A tuple containing:
                (archive_path, flow_class_name).
        """
        parent_directory = self.output_workspace_path.parent
        archive_path = parent_directory / "experiment"

        # Create a ZIP archive of the generated_workspace directory
        arch_path = shutil.make_archive(str(archive_path), "zip", str(self.output_workspace_path))

        print(f"Archive created at {archive_path}.zip")

        return arch_path, self.flow_class_name

    def generate_requirements(self) -> None:
        """Extracts pip libraries from exported python script
        and append in workspace/requirements.txt
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

    def generate_plan_yaml(self, director_fqdn: str = None, tls: bool = False) -> None:
        """Generate the plan.yaml
        Args:
            director_fqdn (str): Fully qualified domain name of the director node.
            tls (bool, optional): Whether to use TLS for the connection.
        """

        # Get the flow_class details
        flow_details = self._extract_flow_details()

        # Get flow_class_name
        self.flow_class_name = flow_details["flow_class_name"]

        # Get flow configuration
        flow_config = self.code_analyzer.fetch_flow_configuration(flow_details)

        # Determine the path for the plan.yaml file
        plan = self.output_workspace_path.joinpath("plan", "plan.yaml").resolve()

        # Initialize the YAML data
        data = self._initialize_plan_yaml(plan)

        # Update the plan_configuration with the analyzed flow configuration
        data["federated_flow"].update(flow_config["federated_flow"])

        # Updating the aggregator address with director's hostname and tls settings in plan.yaml
        if director_fqdn:
            network_settings = Plan.parse(plan).config["network"]
            data["network"] = network_settings
            data["network"]["settings"]["agg_addr"] = director_fqdn
            data["network"]["settings"]["tls"] = tls

        # Write the updated plan configuraiton to the plan.yaml file
        Plan.dump(plan, data)

    def generate_data_yaml(self) -> None:
        """Generate data.yaml"""

        # Get flow class_name
        if not hasattr(self, "flow_class_name"):
            flow_details = self._extract_flow_details()
            self.flow_class_name = flow_details["flow_class_name"]

        # Get runtime information using CodeAnalyzer
        runtime, flow_instance_name = self.code_analyzer.get_flow_runtime_info(self.flow_class_name)

        # Determine the path for the data.yaml
        data_yaml = self.output_workspace_path.joinpath("plan", "data.yaml").resolve()

        # Initialize the YAML data
        data = self._initialize_data_yaml(data_yaml)

        # Initiaize runtime name
        runtime_name = "local_runtime"

        # Process aggregator information using CodeAnalyzer
        runtime_created = self.code_analyzer.process_aggregator(
            runtime, data, flow_instance_name, runtime_name
        )

        # Process collaborator information using CodeAnalyzer
        self.code_analyzer.process_collaborators(
            runtime, data, flow_instance_name, runtime_created, runtime_name
        )

        # Write updated data configuration to the data.yaml file
        Plan.dump(data_yaml, data)

    def _extract_flow_details(self) -> str:
        """Extract the flow class details"""
        flspsec = import_module("openfl.experimental.workflow.interface").FLSpec
        flow_details = self.code_analyzer.get_flow_class_details(flspsec)
        return flow_details

    def _initialize_plan_yaml(self, plan_yaml) -> dict:
        """Load or initialize the plan YAML data.
        Args:
            plan_yaml (Path): The path to the plan.yaml file.

        Returns:
            dict: The data dictionary from plan.yaml.
        """
        data = Plan.load(plan_yaml)
        if data is None:
            data = {}
            data["federated_flow"] = {"settings": {}, "template": ""}
        return data

    def _initialize_data_yaml(self, data_yaml) -> dict:
        """Load or initialize the YAML data.
        Args:
            data_yaml (Path): The path to the data.yaml file.

        Returns:
            dict: The data dictionary from data.yaml
        """
        data = Plan.load(data_yaml)
        return data if data is not None else {}
