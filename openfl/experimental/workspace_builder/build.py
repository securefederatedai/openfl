# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import ast
import astor
from glob import glob
from shutil import copytree
from logging import getLogger
from pathlib import Path

from nbdev.export import nb_export
from openfl.experimental.interface.cli.cli_helper import print_tree


class WorkspaceBuilder:
    def __init__(self, notebook_path: str, ouput_dir: str,
                 template_workspace_path: str) -> None:
        self.__set_logger()

        self.notebook_path = Path(notebook_path).resolve()
        self.output_dir = Path(ouput_dir).resolve()
        self.template_workspace_path = Path(template_workspace_path).resolve()

        # Copy template workspace to output directory
        self.created_workspace_path = Path(copytree(self.template_workspace_path,
                                                    self.output_dir.joinpath(self.notebook_path.name)))
        self.logger.info(f"Copied template workspace to {self.created_workspace_path}")
        print_tree(self.created_workspace_path, level=2)

        self.logger.info("Converting jupter notebook to python script...")
        self.script_path = Path(self.__convert_to_python(self.notebook_path, self.created_workspace_path)).resolve()
        self.script_name = self.script_path.name.split(".")[0].strip()

    def __set_logger(self):
        self.logger = getLogger(__name__)

    def __convert_to_python(self, notebook_path: Path, output_path: Path):
        nb_export(notebook_path, output_path)

        return glob(os.path.join(output_path, "*.py"), recursive=False)[0]

    def __find_flow_classname(self):
        pass

    def generate_requirements(self):
        data = None
        with open(self.script_path, "r") as f:
            requirements = []
            line_nos = []
            data = f.readlines()
            for i, line in enumerate(data):
                line = line.strip()
                if "pip install" in line:
                    line_nos.append(i)
                    # Avoid commented lines, installation from requirements.txt file, or openfl.git
                    # installation
                    if not line.startswith("#") and "-r" not in line and "openfl.git" not in line:
                        requirements.append(f"{line.split(' ')[-1].strip()}\n")

        requirements_filepath = str(self.created_workspace_path.joinpath("requirements.txt").resolve())

        with open(requirements_filepath, "a") as f:
            f.writelines(requirements)

        # Delete pip requirements from python script
        with open(self.script_path, "w") as f:
            for i, line in enumerate(data):
                if i not in line_nos:
                    f.write(line)

    def generate_flow_class(self):
        self.logger.info("Extracting Flow class from python script...")
        self.__flow_class_sourcecode, self.__flow_classname = self.__get_class_artifact("FLSpec")

        if self.__flow_class_sourcecode is None or self.__flow_classname is None:
            self.logger.error("FlowClass not found in extracted script")

        # TODO: Write the flow class in another file? (Not right now)

    def __get_class_artifact(self, parent_classname):
        with open(self.script_path, "r") as s:
            tree = ast.parse(s.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == parent_classname:
                            class_definition = astor.to_source(node)
                            return class_definition, node.name
        return None, None

    def __find_class_initialization(self, script_data, classname, local_vars=None):
        node = ast.parse(script_data)

        if local_vars is None:
            local_vars = {}

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == classname:
            arguments = {}

            for arg_node in node.args:
                try:
                    arg_value = eval(compile(ast.Expression(arg_node),
                                    filename="", mode="eval"), local_vars)
                    arg_name = ast.dump(arg_node, annotate_fields=False,
                                        include_attributes=False)
                    arguments[arg_name] = arg_value
                except Exception as e:
                    arguments[arg_name] = self.__find_objects(script_data, arg_name)

            for keyword_node in node.keywords:
                try:
                    keyword_value = eval(compile(ast.Expression(keyword_node.value),
                                        filename="", mode="eval"), local_vars)
                    arguments[keyword_node.arg] = keyword_value
                except Exception as e:
                    arguments[keyword_node.arg] = self.__find_objects(script_data, keyword_node.arg)

            return arguments

        for child_node in ast.iter_child_nodes(node):
            if isinstance(child_node, ast.FunctionDef):
                local_vars.update({arg.arg: None for arg in child_node.args.args})
            result = self.__find_class_initialization(child_node, classname, local_vars)

            return result if result else None

    def __find_objects(self, script_data, object_name):
        node = ast.parse(script_data)

        for child_node in ast.iter_child_nodes(node):
            if hasattr(child_node, "name") and child_node.name == object_name:
                return astor.to_source(child_node)

        return None

    def generate_data_yaml(self):
        pass

    def generate_plan_yaml(self):
        if not hasattr(self, "__flow_classname"):
            self.generate_flow_class()

        with open(self.script_path, "r") as f:
            class_arguments = self.__find_class_initialization(f.read(), self.__flow_classname)

        print("\nFilling federated_flow details in plan.yaml...")
        with open(self.created_workspace_path.joinpath("plan").joinpath("plan.yaml"), "r") as f:
            data = yaml.safe_load(f)
    
        with open(self.created_workspace_path.joinpath("plan").joinpath("plan.yaml"), "w") as f:
            data["federated_flow"]["template"] = f"src.{self.script_name}.{self.__flow_classname}"
            data["federated_flow"]["settings"] = class_arguments
            yaml.safe_dump(data, f)

    def generate_workspace_scripts(self):
        pass
