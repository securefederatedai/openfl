# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Notebook Tools module."""

import shutil
from importlib import import_module
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, Tuple

from openfl.experimental.workflow.federated.plan import Plan
from openfl.experimental.workflow.interface.cli.cli_helper import WORKSPACE, print_tree
from openfl.experimental.workflow.notebooktools.code_analyzer import CodeAnalyzer


class NotebookTools:
    """Class providing utility functions to convert Jupyter notebook based on Workflow API
    into a workspace enabling its deployment on distributed infrastructure

    Attributes:
        notebook_path (Path): Path to Jupyter notebook to be converted.
        output_workspace_path (Path): Target directory for generated workspace.
        template_workspace_path (Path): Path to template workspace provided with
            OpenFL.
        code_analyzer (CodeAnalyzer): An instance of the CodeAnalyzer class for analyzing
            notebook code.
        flow_class_name (str): Name of the flow class.
    """

    def __init__(self, notebook_path: str, output_workspace: str) -> None:
        """Initialize a NotebookTools object.
        Args:
            notebook_path (str): Path to Jupyter notebook to be converted.
            output_workspace (str): Target directory for generated workspace
        """
        self.notebook_path = Path(notebook_path).resolve()
        if not self.notebook_path.exists() or not self.notebook_path.is_file():
            raise FileNotFoundError(f"The Jupyter notebook at {notebook_path} does not exist.")

        self.output_workspace_path = Path(output_workspace).resolve()
        self._initialize_workspace()

        self.code_analyzer = CodeAnalyzer(self.notebook_path, self.output_workspace_path)

    def _initialize_workspace(self) -> None:
        """Initialize the workspace by setting up path and copying templates"""
        # Regenerate the workspace if it already exists
        if self.output_workspace_path.exists():
            print(f"Removing existing workspace: {self.output_workspace_path}")
            shutil.rmtree(self.output_workspace_path)
        self.output_workspace_path.parent.mkdir(parents=True, exist_ok=True)

        self.template_workspace_path = (WORKSPACE / "template_workspace").resolve(strict=True)
        copytree(self.template_workspace_path, self.output_workspace_path)
        print(f"Copied template workspace to {self.output_workspace_path}")

    def _generate_experiment_archive(self) -> Tuple[str, str]:
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

    def _generate_requirements(self) -> None:
        """Extract dependencies (pip install <module name>) from exported python script
        and append to workspace/requirements.txt
        """
        try:
            requirements, requirements_line_numbers, data = self.code_analyzer.get_requirements()
            requirements_filepath = str(
                self.output_workspace_path.joinpath("requirements.txt").resolve()
            )
            with open(requirements_filepath, "a") as f:
                f.writelines(requirements)
            self.code_analyzer.remove_lines(data, requirements_line_numbers)

            print(f"Successfully generated {requirements_filepath}")

        except Exception as e:
            print(f"Failed to generate requirements: {e}")

    def _generate_plan_yaml(self, director_fqdn: str = None, tls: bool = False) -> None:
        """Generates workspace/plan.yaml
        Args:
            director_fqdn (str): Fully qualified domain name of the director node.
            tls (bool, optional): Whether to use TLS for the connection.
        """
        flow_details = self._extract_flow_details()
        flow_config = self.code_analyzer.fetch_flow_configuration(flow_details)
        plan_path = self.output_workspace_path.joinpath("plan", "plan.yaml").resolve()
        data_config = self._build_plan_config(flow_config, director_fqdn, tls, plan_path)
        Plan.dump(plan_path, data_config)

    def _build_plan_config(
        self, flow_config: Dict[str, Any], director_fqdn: str, tls: bool, plan_path: Path
    ) -> Dict[str, Any]:
        """
        Build plan configuration with validation.

        Args:
            flow_config: Flow configuration dictionary
            director_fqdn: Director's FQDN
            tls: TLS setting
            plan_path: Path to plan.yaml

        Returns:
            Dict[str, Any]: Complete plan configuration
        """
        data_config = self._initialize_plan_yaml(plan_path)
        data_config["federated_flow"].update(flow_config["federated_flow"])
        if director_fqdn:
            network_settings = Plan.parse(plan_path).config["network"]
            data_config["network"] = network_settings
            data_config["network"]["settings"]["agg_addr"] = director_fqdn
            data_config["network"]["settings"]["tls"] = tls

        return data_config

    def _extract_flow_details(self) -> str:
        """Extract the flow class details"""
        flspsec = import_module("openfl.experimental.workflow.interface").FLSpec
        flow_details = self.code_analyzer.get_flow_class_details(flspsec)
        # Set flow_class_name for future reference
        self.flow_class_name = flow_details["flow_class_name"]

        return flow_details

    def _initialize_plan_yaml(self, plan_yaml: Path) -> dict:
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

    def export(self, director_fqdn: str, tls: bool = False) -> Tuple[str, str]:
        """Exports workspace for FederatedRuntime.
        Args:
            director_fqdn (str): Fully qualified domain name of the director node.
            tls (bool, optional): Whether to use TLS for the connection.

        Returns:
            Tuple[str, str]: A tuple containing:
                (archive_path, flow_class_name).
        """
        self._generate_requirements()
        self._generate_plan_yaml(director_fqdn, tls)
        print_tree(self.output_workspace_path, level=2)

        return self._generate_experiment_archive()
