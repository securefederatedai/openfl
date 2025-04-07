# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
from nbdev.export import nb_export


class CodeAnalyzer:
    """Analyzes and process Jupyter Notebooks.
      Provides code extraction and transformation functionality

    Attributes:
        script_path (Path): Absolute path to the python script generated.
        script_name (str): Name of the generated python script.
        exported_script_module (ModuleType): The imported module object of the generated script.
        available_modules_in_exported_script (list): List of available attributes in the
            exported script.
    """

    def __init__(self, notebook_path: Path, output_path: Path) -> None:
        """Initialize CodeAnalyzer and process the script from notebook

        Args:
            notebook_path (Path): Path to Jupyter notebook to be converted.
            output_path (Path): The directory where the converted Python script will be saved.
        """
        print("Converting jupyter notebook to python script...")
        # Extract the export filename from the notebook
        self.script_name = self.__get_exp_name(notebook_path)
        # Convert the notebook to a Python script and set the script path
        self.script_path = Path(
            self.__convert_to_python(
                notebook_path,
                output_path.joinpath("src"),
                f"{self.script_name}.py",
            )
        ).resolve()
        self.__comment_flow_execution()

    def __get_exp_name(self, notebook_path: Path) -> str:
        """Extract experiment name from Jupyter notebook
        Looks for '#| default_exp <name>' pattern in code cells
        and extracts the experiment name. The name must be a valid Python identifier.

        Args:
            notebook_path (str): Path to Jupyter notebook.
        """
        with notebook_path.open("r") as f:
            notebook_content = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        for cell in notebook_content.cells:
            if cell.cell_type == "code":
                code = cell.source
                match = re.search(r"#\s*\|\s*default_exp\s+(\w+)", code)
                if match:
                    print(f"Retrieved {match.group(1)} from default_exp")
                    return match.group(1)
        raise ValueError(
            "The notebook does not contain a '#| default_exp <experiment_name' marker."
            "Please add the marker to the first cell of the notebook"
        )

    def __convert_to_python(self, notebook_path: Path, output_path: Path, export_filename) -> Path:
        """Converts a Jupyter notebook to a Python script.
        Args:
            notebook_path (Path): The path to the Jupyter notebook file
                to be converted.
            output_path (Path): The directory where the exported Python
                script should be saved.
            export_filename: The name of the exported Python script file.

        Returns:
            Path: The path to the exported Python script file.
        """
        nb_export(notebook_path, output_path)

        return Path(output_path).joinpath(export_filename).resolve()

    def __comment_flow_execution(self) -> None:
        """Comment out lines containing '.run()' in the specified Python script"""
        run_statement = ".run()"

        with self.script_path.open("r") as f:
            data = f.readlines()
        for idx, line in enumerate(data):
            if run_statement in line:
                data[idx] = f"# {line}"
        with self.script_path.open("w") as f:
            f.writelines(data)

    def __import_generated_script(self) -> None:
        """
        Imports the generated python script using the importlib module
        """
        try:
            sys.path.append(str(self.script_path.parent))
            self.exported_script_module = import_module(self.script_name)
            self.available_modules_in_exported_script = dir(self.exported_script_module)
        except ImportError as e:
            raise ImportError(f"Failed to import script {self.script_name}: {e}")

    def __get_class_arguments(self, class_name) -> list:
        """Given the class name returns expected class arguments.

        Args:
            class_name (str): The name of the class.

        Returns:
            list: A list of expected class arguments.
        """
        if not hasattr(self, "exported_script_module"):
            self.__import_generated_script()

        # Find class from imported python script module
        for idx, attr in enumerate(self.available_modules_in_exported_script):
            if attr == class_name:
                cls = getattr(
                    self.exported_script_module,
                    self.available_modules_in_exported_script[idx],
                )
        if "cls" not in locals():
            raise NameError(f"{class_name} not found.")

        if inspect.isclass(cls):
            if "__init__" in cls.__dict__:
                init_signature = inspect.signature(cls.__init__)
                # Extract the parameter names (excluding 'self', 'args', and
                # 'kwargs')
                arg_names = [
                    param
                    for param in init_signature.parameters
                    if param not in ("self", "args", "kwargs")
                ]
                return arg_names
            return []
        print(f"{cls} is not a class")

    def __get_class_name(self, parent_class) -> Optional[str]:
        """Find and return the name of a class derived from the provided parent class.
        Args:
            parent_class: FLSpec instance.

        Returns:
            Optional[str]: The name of the derived class.
        """
        if not hasattr(self, "exported_script_module"):
            self.__import_generated_script()

        # Going though all attributes in imported python script
        for attr in self.available_modules_in_exported_script:
            t = getattr(self.exported_script_module, attr)
            if inspect.isclass(t) and t != parent_class and issubclass(t, parent_class):
                return attr
        raise ValueError("No flow class found that inherits from FLSpec")

    def __extract_class_initializing_args(self, class_name) -> Dict[str, Any]:
        """Provided name of the class returns expected arguments and it's
        values in form of dictionary.
        Args:
            class_name (str): The name of the class.

        Returns:
            Dict[str, Any]: A dictionary containing the expected arguments and their values.
        """
        instantiation_args = {"args": {}, "kwargs": {}}

        with open(self.script_path, "r") as s:
            tree = ast.parse(s.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == class_name:
                        # We found an instantiation of the class
                        instantiation_args["args"] = self._extract_positional_args(node.args)
                        instantiation_args["kwargs"] = self._extract_keyword_args(node.keywords)

        return instantiation_args

    def _extract_positional_args(self, args) -> Dict[str, Any]:
        """Extract positional arguments from the AST nodes.
        Args:
            args: AST nodes representing the arguments.

        Returns:
            Dict[str, Any]: Dictionary of argument names and their values.
        """
        positional_args = {}
        for arg in args:
            if isinstance(arg, ast.Name):
                positional_args[arg.id] = arg.id
            elif isinstance(arg, ast.Constant):
                positional_args[arg.s] = ast.unparse(arg)
            else:
                positional_args[arg.arg] = ast.unparse(arg).strip()
        return positional_args

    def _extract_keyword_args(self, keywords) -> Dict[str, Any]:
        """Extract keyword arguments from the AST nodes.
        Args:
            keywords: AST nodes representing the keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of keyword argument names and their values.
        """
        keyword_args = {}
        for kwarg in keywords:
            value = ast.unparse(kwarg.value).strip()
            value = self._clean_value(value)
            try:
                value = ast.literal_eval(value)
            except ValueError:
                pass
            keyword_args[kwarg.arg] = value
        return keyword_args

    def _clean_value(self, value: str) -> str:
        """Clean the value by removing unnecessary parentheses or brackets.
        Args:
            value (str): The string value to be cleaned.

        Returns:
            str: The cleaned string value
        """
        if value.startswith("(") and "," not in value:
            value = value.lstrip("(").rstrip(")")
        if value.startswith("[") and "," not in value:
            value = value.lstrip("[").rstrip("]")
        return value

    def get_requirements(self) -> Tuple[List[str], List[int], List[str]]:
        """Extract pip libraries from the script

        Returns:
            tuple: A tuple containing:
                requirements (list of str): List of pip libraries found in the script.
                line_nos (list of int): List of line numbers where "pip install" commands are found.
                data (list of str): The entire script data as a list of lines.
        """
        data = None
        with self.script_path.open("r") as f:
            requirements = []
            line_nos = []
            data = f.readlines()
            for i, line in enumerate(data):
                line = line.strip()
                if "pip install" in line:
                    line_nos.append(i)
                    # Avoid commented lines, libraries from *.txt file, or openfl.git
                    # installation
                    if not line.startswith("#") and "-r" not in line and "openfl.git" not in line:
                        requirements.append(f"{line.split(' ')[-1].strip()}\n")

            return requirements, line_nos, data

    def remove_lines(self, data: List[str], line_nos: List[int]) -> None:
        """Removes pip install lines from the script
        Args:
            data (List[str]): The entire script data as a list of lines.
            line_nos (List[int]): List of line numbers where "pip install" commands are found.
        """
        with self.script_path.open("w") as f:
            for i, line in enumerate(data):
                if i not in line_nos:
                    f.write(line)

    def get_flow_class_details(self, parent_class) -> Dict[str, Any]:
        """
        Retrieves details of a flow class that inherits from the given parent clas
        Args:
            parent_class: The parent class (FLSpec instance).

        Returns:
            Dict[str, Any]: A dictionary containing:
                flow_class_name (str): The name of the flow class.
                expected_args (List[str]): The expected arguments for the flow class.
                init_args (Dict[str, Any]): The initialization arguments for the flow class.
        """
        flow_class_name = self.__get_class_name(parent_class)
        expected_arguments = self.__get_class_arguments(flow_class_name)
        init_args = self.__extract_class_initializing_args(flow_class_name)

        return {
            "flow_class_name": flow_class_name,
            "expected_args": expected_arguments,
            "init_args": init_args,
        }

    def fetch_flow_configuration(self, flow_details: Dict[str, Any]) -> Dict[str, Any]:
        """Get flow configuration from flow details.
        Args:
            flow_details (Dict[str, Any]): Dictionary containing flow class details.

        Returns:
            Dict[str, Any]: Dictionary containing the plan configuration
        """
        flow_config = {
            "federated_flow": {
                "settings": {},
                "template": f"src.{self.script_name}.{flow_details['flow_class_name']}",
            }
        }

        def update_dictionary(args: dict, dtype: str = "args") -> None:
            """Update plan configuration with argument values.

            Args:
                args: Dictionary of arguments to process
                dtype: Type of arguments ('args' or 'kwargs')
            """
            for idx, (k, v) in enumerate(args.items()):
                if dtype == "args":
                    v = getattr(self.exported_script_module, str(k), None)
                    if v is not None and not isinstance(v, (int, str, bool)):
                        v = f"src.{self.script_name}.{k}"
                    k = flow_details["expected_args"][idx]
                elif dtype == "kwargs":
                    if v is not None and not isinstance(v, (int, str, bool)):
                        v = f"src.{self.script_name}.{v}"
                flow_config["federated_flow"]["settings"].update({k: v})

        # Process arguments
        pos_args = flow_details["init_args"].get("args", {})
        update_dictionary(pos_args, "args")
        kw_args = flow_details["init_args"].get("kwargs", {})
        update_dictionary(kw_args, "kwargs")

        return flow_config
