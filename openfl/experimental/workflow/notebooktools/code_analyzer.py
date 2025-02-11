# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import re
import sys
from importlib import import_module
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
from nbdev.export import nb_export

logger = getLogger(__name__)


class CodeAnalyzer:
    """Code analysis and transformation functionality for NotebookTools

    Attributes:
       script_path: Absolute path to python script.
       script_name: Name of the python script.
    """

    def __init__(self, notebook_path: Path, output_path: Path) -> None:
        """Initialize CodeAnalzer and process the script from notebook

        Args:
            notebook_path (Path): The path to the Jupyter notebook that needs to be converted.
            output_path (Path): The directory where the converted Python script will be saved.
        """
        logger.info("Converting jupter notebook to python script...")

        # Extract the export filename from the notebook
        export_filename = self.__get_exp_name(notebook_path)
        if export_filename is None:
            raise NameError(
                "Please include `#| default_exp <experiment_name>` in "
                "the first cell of the notebook."
            )
        # Convert the notebook to a Python script and set the script path
        self.script_path = Path(
            self.__convert_to_python(
                notebook_path,
                output_path.joinpath("src"),
                f"{export_filename}.py",
            )
        ).resolve()
        # Generated python script name
        self.script_name = self.script_path.name.split(".")[0].strip()

        # Comment out flow.run() to prevent the flow from starting execution
        # automatically when the script is imported.
        self.__comment_flow_execution()

        # Change the runtime backend from 'ray' to 'single_process'
        self.__change_runtime()

    def __get_exp_name(self, notebook_path: Path) -> None:
        """Fetch the experiment name from the Jupyter notebook.
        Args:
            notebook_path (str): Path to Jupyter notebook.
        """
        with open(str(notebook_path), "r") as f:
            notebook_content = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        for cell in notebook_content.cells:
            if cell.cell_type == "code":
                code = cell.source
                match = re.search(r"#\s*\|\s*default_exp\s+(\w+)", code)
                if match:
                    logger.info(f"Retrieved {match.group(1)} from default_exp")
                    return match.group(1)
        return None

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
        with open(self.script_path, "r") as f:
            data = f.readlines()
        for idx, line in enumerate(data):
            if ".run()" in line:
                data[idx] = f"# {line}"
        with open(self.script_path, "w") as f:
            f.writelines(data)

    def __change_runtime(self) -> None:
        """Change the LocalRuntime backend from ray to single_process."""
        with open(self.script_path, "r") as f:
            data = f.read()

        if "backend='ray'" in data or 'backend="ray"' in data:
            data = data.replace("backend='ray'", "backend='single_process'").replace(
                'backend="ray"', 'backend="single_process"'
            )

        with open(self.script_path, "w") as f:
            f.write(data)

    def __import_exported_script(self) -> None:
        """
        Imports the generated python script using the importlib module
        """
        try:
            sys.path.append(str(self.script_path.parent))
            self.exported_script_module = import_module(self.script_name)
            self.available_modules_in_exported_script = dir(self.exported_script_module)

        except ImportError as e:
            logger.error(f"Failed to import script {self.script_name}: {e}")
            raise

    def __get_class_arguments(self, class_name) -> list:
        """Given the class name returns expected class arguments.

        Args:
            class_name (str): The name of the class.

        Returns:
            list: A list of expected class arguments.
        """
        # Import python script if not already
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        # Find class from imported python script module
        for idx, attr in enumerate(self.available_modules_in_exported_script):
            if attr == class_name:
                cls = getattr(
                    self.exported_script_module,
                    self.available_modules_in_exported_script[idx],
                )

        # If class not found
        if "cls" not in locals():
            raise NameError(f"{class_name} not found.")

        if inspect.isclass(cls):
            # Check if the class has an __init__ method
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
        logger.error(f"{cls} is not a class")

    def __get_class_name_and_sourcecode_from_parent_class(
        self, parent_class
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """Provided the parent_class name returns derived class source code and
        name.
        Args:
            parent_class: FLSpec instance.

        Returns:
            Optional[Tuple[Optional[str], Optional[str]]]:
                The source code of the derived class (str).
                The name of the derived class (str).
        """
        # Import python script if not already
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        # Going though all attributes in imported python script
        for attr in self.available_modules_in_exported_script:
            t = getattr(self.exported_script_module, attr)
            if inspect.isclass(t) and t != parent_class and issubclass(t, parent_class):
                return inspect.getsource(t), attr

        return None, None

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
        """Extract positional arguments from the AST nodes."""
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
        """Extract keyword arguments from the AST nodes."""
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
        """Clean the value by removing unnecessary parentheses or brackets."""
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
        with open(self.script_path, "r") as f:
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
        with open(self.script_path, "w") as f:
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
        _, flow_class_name = self.__get_class_name_and_sourcecode_from_parent_class(parent_class)
        if not flow_class_name:
            raise ValueError("No flow class found that inherits from FLSpec")

        # Get expected arguments
        expected_arguments = self.__get_class_arguments(flow_class_name)

        # get initialization arguments
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

    def get_flow_runtime_info(self, flow_class_name: str) -> Tuple[object, str]:
        """Get federated flow class and runtime information.
        Args:
            flow_class_name (str): The name of the federated flow class to retrieve.

        Returns:
            tuple: A tuple containing the runtime instance and the flow class name.
        """
        if not hasattr(self, "exported_script_module"):
            self.__import_exported_script()

        federated_flow_class = getattr(self.exported_script_module, flow_class_name)
        flow_instance_name, runtime = self._find_flow_instance_runtime(federated_flow_class)
        return runtime, flow_instance_name

    def _find_flow_instance_runtime(self, federated_flow_class) -> Tuple[str, object]:
        """Find runtime instance
        Args:
            federated_flow_class: The class object of the federated flow.

        Returns:
            tuple: A tuple containing the name of the flow instance and the runtime instance.
        """
        for t in self.available_modules_in_exported_script:
            tempstring = t
            t = getattr(self.exported_script_module, t)
            if isinstance(t, federated_flow_class):
                flow_instance_name = tempstring
                if not hasattr(t, "_runtime"):
                    raise AttributeError("Unable to locate LocalRuntime instantiation")
                runtime = t._runtime
                if not hasattr(runtime, "collaborators"):
                    raise AttributeError("LocalRuntime instance does not have collaborators")
                return flow_instance_name, runtime
        raise AttributeError("Runtime instance not found")

    def process_aggregator(self, runtime, data, flow_instance_name, runtime_name) -> bool:
        """Process the aggregator details.
        Args:
            runtime (Any): The runtime instance containing the aggregator.
            data (Dict[str, Any]): The data dictionary to be updated with aggregator details.
            flow_instance_name (str): The name of the flow instance.
            runtime_name (str): The name of the runtime.

        Returns:
            bool: A boolean indicating whether the runtime was created.
        """
        aggregator = runtime._aggregator
        runtime_created = False
        private_attrs_callable = aggregator.private_attributes_callable
        aggregator_private_attributes = aggregator.private_attributes

        if private_attrs_callable is not None:
            data["aggregator"] = {
                "callable_func": {
                    "settings": {},
                    "template": f"src.{self.script_name}.{private_attrs_callable.__name__}",
                }
            }
            arguments_passed_to_initialize = self.__extract_class_initializing_args("Aggregator")[
                "kwargs"
            ]
            agg_kwargs = aggregator.kwargs
            for key, value in agg_kwargs.items():
                if isinstance(value, (int, str, bool)):
                    data["aggregator"]["callable_func"]["settings"][key] = value
                else:
                    arg = arguments_passed_to_initialize[key]
                    value = f"src.{self.script_name}.{arg}"
                    data["aggregator"]["callable_func"]["settings"][key] = value
        elif aggregator_private_attributes:
            runtime_created = True
            with open(self.script_path, "a") as f:
                f.write(f"\n{runtime_name} = {flow_instance_name}._runtime\n")
                f.write(
                    f"\naggregator_private_attributes = "
                    f"{runtime_name}._aggregator.private_attributes\n"
                )
            data["aggregator"] = {
                "private_attributes": f"src.{self.script_name}.aggregator_private_attributes"
            }
        return runtime_created

    def process_collaborators(
        self, runtime, data, flow_instance_name, runtime_created, runtime_name
    ) -> Dict[str, Any]:
        """Process the collaborators.
        Args:
            runtime (Any): The runtime instance containing the collaborators.
            data (Dict[str, Any]): The data dictionary to be updated with collaborator details.
            flow_instance_name (str): The name of the flow instance.
            runtime_created (bool): Flag indicating if the runtime has been created.
            runtime_name (str): The name of the runtime.

        Returns:
            Dict[str, Any]: The updated data dictionary with collaborator details.
        """
        collaborators = runtime._LocalRuntime__collaborators
        arguments_passed_to_initialize = self.__extract_class_initializing_args("Collaborator")[
            "kwargs"
        ]
        runtime_collab_created = False

        for collab in collaborators.values():
            collab_name = collab.get_name()
            callable_func = collab.private_attributes_callable
            private_attributes = collab.private_attributes

            if callable_func:
                if collab_name not in data:
                    data[collab_name] = {"callable_func": {"settings": {}, "template": None}}
                kw_args = runtime.get_collaborator_kwargs(collab_name)
                for key, value in kw_args.items():
                    if key == "private_attributes_callable":
                        value = f"src.{self.script_name}.{value}"
                        data[collab_name]["callable_func"]["template"] = value
                    elif isinstance(value, (int, str, bool)):
                        data[collab_name]["callable_func"]["settings"][key] = value
                    else:
                        arg = arguments_passed_to_initialize[key]
                        value = f"src.{self.script_name}.{arg}"
                        data[collab_name]["callable_func"]["settings"][key] = value
            elif private_attributes:
                with open(self.script_path, "a") as f:
                    if not runtime_created:
                        f.write(f"\n{runtime_name} = {flow_instance_name}._runtime\n")
                        runtime_created = True
                    if not runtime_collab_created:
                        f.write(
                            f"\nruntime_collaborators = {runtime_name}._LocalRuntime__collaborators"
                        )
                        runtime_collab_created = True
                    f.write(
                        f"\n{collab_name}_private_attributes = "
                        f"runtime_collaborators['{collab_name}'].private_attributes"
                    )
                data[collab_name] = {
                    "private_attributes": f"src.{self.script_name}.{collab_name}_private_attributes"
                }
