# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import ast
import astor
import inspect
import importlib

from typing import Any
from shutil import copytree
from logging import getLogger
from pathlib import Path

from nbdev.export import nb_export
from openfl.experimental.interface.cli.cli_helper import print_tree


class WorkspaceBuilder:
    def __init__(self, notebook_path: str, export_filename: str, ouput_dir: str,
                 template_workspace_path: str) -> None:
        self.logger = getLogger(__name__)

        self.notebook_path = Path(notebook_path).resolve()
        self.output_dir = Path(ouput_dir).resolve()
        self.template_workspace_path = Path(template_workspace_path).resolve()

        # Copy template workspace to output directory
        self.created_workspace_path = Path(copytree(self.template_workspace_path,
                                                    self.output_dir.joinpath(self.notebook_path.name)))
        self.logger.info(f"Copied template workspace to {self.created_workspace_path}")
        print_tree(self.created_workspace_path, level=2)

        self.logger.info("Converting jupter notebook to python script...")
        self.script_path = Path(self.__convert_to_python(
            self.notebook_path, self.created_workspace_path.joinpath("src"),
            f"{export_filename}.py")).resolve()

        self.script_name = self.script_path.name.split(".")[0].strip()
        self.__comment_flow_execution()
        # This is required as Ray created actors too many actors when backend="ray"
        self.__change_runtime()

    def __convert_to_python(self, notebook_path: Path, output_path: Path, export_filename):
        nb_export(notebook_path, output_path)

        return Path(output_path).joinpath(export_filename).resolve()

    def __comment_flow_execution(self):
        with open(self.script_path, "r+") as f:
            data = f.readlines()
        for idx, line in enumerate(data):
            if ".run()" in line:
                data[idx] = f"# {line}"
        with open(self.script_path, "w") as f:
            f.writelines(data)

    def __change_runtime(self):
        with open(self.script_path, "r") as f:
            data = f.read()

        if data.find("backend='ray'") != -1:
            data = data.replace("backend='ray'", "backend='single_process'")
        elif data.find('backend="ray"') != -1:
            data = data.replace('backend="ray"', 'backend="single_process"')

        with open(self.script_path, "w") as f:
            f.write(data)

    def __get_class_arguments(self, class_name):
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        for idx, attr in enumerate(self.available_modules_in_exported_script):
            if attr == class_name:
                cls = getattr(self.exported_script_module,
                              self.available_modules_in_exported_script[idx])

        if "cls" not in locals():
            raise Exception(f"{class_name} not found.")

        if inspect.isclass(cls):
            # Check if the class has an __init__ method
            if "__init__" in cls.__dict__:
                init_signature = inspect.signature(cls.__init__)
                # Extract the parameter names (excluding 'self')
                arg_names = [param for param in init_signature.parameters if param not in (
                    "self", "args", "kwargs")]
                return arg_names
        print(f"{cls} is not a class")

    def __get_class_name_and_sourcecode_from_parent_class(self, parent_class):
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        for attr in self.available_modules_in_exported_script:
            t = getattr(self.exported_script_module, attr)
            if inspect.isclass(t) and t != parent_class and issubclass(t, parent_class):
                return inspect.getsource(t), attr

        return None, None

    def __extract_class_initializing_args(self, class_name):
        instantiation_args = {
            "args": {}, "kwargs": {}
        }

        with open(self.script_path, "r") as s:
            tree = ast.parse(s.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == class_name:
                    # We found an instantiation of the class
                    for arg in node.args:
                        # Iterate through positional arguments
                        if isinstance(arg, ast.Name):
                            # Use the variable name as the argument value
                            instantiation_args["args"][arg.id] = arg.id
                        elif isinstance(arg, ast.Constant):
                            instantiation_args["args"][arg.s] = astor.to_source(arg)
                        else:
                            instantiation_args["args"][arg.arg] = astor.to_source(arg).strip()

                    for kwarg in node.keywords:
                        # Iterate through keyword arguments
                        value = astor.to_source(kwarg.value).strip()
                        if value.startswith("(") and "," not in value:
                            value = value.lstrip("(").rstrip(")")
                        if value.startswith("[") and "," not in value:
                            value = value.lstrip("[").rstrip("]")
                        try:
                            value = ast.literal_eval(value)
                        except Exception:
                            pass
                        instantiation_args["kwargs"][kwarg.arg] = value

        return instantiation_args

    def __import_exported_script(self):
        import importlib

        current_dir = os.getcwd()
        os.chdir(self.script_path.parent)
        self.exported_script_module = importlib.import_module(self.script_name)
        self.available_modules_in_exported_script = dir(self.exported_script_module)
        os.chdir(current_dir)

    def __read_yaml(self, path):
        with open(path, "r") as y:
            return yaml.safe_load(y)

    def __write_yaml(self, path, data):
        with open(path, "w") as y:
            yaml.safe_dump(data, y)

    # Have to do generate_requirements before anything else
    # because these !pip commands needs to be removed from python script
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

        requirements_filepath = str(
            self.created_workspace_path.joinpath("requirements.txt").resolve())

        with open(requirements_filepath, "a") as f:
            f.writelines(requirements)

        # Delete pip requirements from python script
        with open(self.script_path, "w") as f:
            for i, line in enumerate(data):
                if i not in line_nos:
                    f.write(line)

    def generate_plan_yaml(self):
        flspec = getattr(
            importlib.import_module("openfl.experimental.interface"), "FLSpec"
        )
        _, self.flow_class_name = self.__get_class_name_and_sourcecode_from_parent_class(flspec)
        self.flow_class_expected_arguments = self.__get_class_arguments(self.flow_class_name)
        self.arguments_passed_to_initialize = self.__extract_class_initializing_args(
            self.flow_class_name)

        plan = self.created_workspace_path.joinpath("plan", "plan.yaml").resolve()
        data = self.__read_yaml(plan)
        if data == None:
            data["federated_flow"] = {
                "settings": {},
                "template": ""
            }

        data["federated_flow"]["template"] = f"src.{self.script_name}.{self.flow_class_name}"

        def update_dictionary(args: dict, data: dict, dtype: str = "args"):
            for idx, (k, v) in enumerate(args.items()):
                if dtype == "args":
                    v = getattr(self.exported_script_module, str(k), None)
                    if v != None and type(v) not in (int, str, bool):
                        v = f"src.{self.script_name}.{k}"
                    k = self.flow_class_expected_arguments[idx]
                elif dtype == "kwargs":
                    if v != None and type(v) not in (int, str, bool):
                        v = f"src.{self.script_name}.{k}"
                data["federated_flow"]["settings"].update({
                    k: v
                })

        pos_args = self.arguments_passed_to_initialize["args"]
        update_dictionary(pos_args, data, dtype="args")
        kw_args = self.arguments_passed_to_initialize["kwargs"]
        update_dictionary(kw_args, data, dtype="kwargs")

        self.__write_yaml(plan, data)

    def generate_data_yaml(self):
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        if not hasattr(self, "flow_class_name"):
            flspec = getattr(
                importlib.import_module("openfl.experimental.interface"), "FLSpec"
            )
            _, self.flow_class_name = self.__get_class_name_and_sourcecode_from_parent_class(
                flspec)

        federated_flow_class = getattr(self.exported_script_module, self.flow_class_name)
        # Find federated_flow._runtime.collaborators
        for t in self.available_modules_in_exported_script:
            t = getattr(self.exported_script_module, t)
            if isinstance(t, federated_flow_class):
                collaborators_names = t._runtime.collaborators
                break

        if "collaborators_names" not in locals():
            raise Exception(f"Unable to find {self.flow_class_name} instance")

        data_yaml = self.created_workspace_path.joinpath("plan", "data.yaml").resolve()
        data = self.__read_yaml(data_yaml)
        if data is None:
            data = {}

        for collab_name in collaborators_names:
            data[collab_name] = {
                "callable_func": {
                    "settings": {},
                    "template": ""
                }
            }

        def update_dictionary(args: dict, data: dict, expected_args: Any, args_passed_to_initialize: Any, component: str = "aggregator", dtype: str = "args"):
            for idx, (k, v) in enumerate(args.items()):
                if k in ("name", "num_cpus", "num_gpus"):
                    continue
                if dtype == "args":
                    v = getattr(self.exported_script_module, k, None)
                    if v != None and not isinstance(v, (int, str, bool)):
                        v = f"src.{self.script_name}.{k}"
                    k = expected_args[idx]
                elif dtype == "kwargs":
                    if v != None and not isinstance(v, (int, bool)) and v in dir(self.exported_script_module):
                        v = getattr(self.exported_script_module,
                                    args_passed_to_initialize["kwargs"][k])
                        if not isinstance(v, (int, str, bool)):
                            v = f"src.{self.script_name}.{args_passed_to_initialize['kwargs'][k]}"
                    elif v != None and type(v) not in (int, bool) and v not in dir(self.exported_script_module):
                        print(f"WARNING: This needs to double checked by user: {k, v}")
                if k == "private_attributes_callable":
                    data[component]["callable_func"]["template"] = v
                else:
                    data[component]["callable_func"]["settings"].update({
                        k: v
                    })

        self.aggregator_expected_arguments = self.__get_class_arguments("Aggregator")
        self.arguments_passed_to_initialize = self.__extract_class_initializing_args("Aggregator")

        pos_args = self.arguments_passed_to_initialize["args"]
        if len(pos_args) > 0:
            if "aggregator" not in data:
                data["aggregator"] = {
                    "callable_func": {
                        "settings": {},
                        "template": ""
                    }
                }
            update_dictionary(pos_args, data, self.aggregator_expected_arguments, None,
                              "aggregator", dtype="args")

        kw_args = self.arguments_passed_to_initialize["kwargs"]
        if len(kw_args) > 0:
            if "aggregator" not in data:
                data["aggregator"] = {
                    "callable_func": {
                        "settings": {},
                        "template": ""
                    }
                }
            update_dictionary(kw_args, data, self.aggregator_expected_arguments,
                              self.arguments_passed_to_initialize, "aggregator", dtype="kwargs")

        self.collaborator_expected_arguments = self.__get_class_arguments("Collaborator")
        self.arguments_passed_to_initialize = self.__extract_class_initializing_args(
            "Collaborator")

        pos_args = self.arguments_passed_to_initialize["args"]
        kw_args = self.arguments_passed_to_initialize["kwargs"]

        for collab_name in collaborators_names:
            if len(pos_args) > 0:
                update_dictionary(pos_args, data, self.collaborator_expected_arguments, None,
                                  collab_name, dtype="args")

            if len(kw_args) > 0:
                update_dictionary(kw_args, data, self.collaborator_expected_arguments,
                                  self.arguments_passed_to_initialize, collab_name, dtype="kwargs")

        self.__write_yaml(data_yaml, data)
