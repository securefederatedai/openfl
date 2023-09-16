# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import ast
import yaml
from pathlib import Path
from shutil import copytree
from glob import glob


def convert_to_python(notebook_path: str, output_path: str):
    from nbdev.export import nb_export
    nb_export(notebook_path, output_path)

    return glob(os.path.join(output_path, "*.py"), recursive=False)[0]


def build_tags_lib(script_path: Path, tags: dict):
    with open(script_path, "r") as f:
        for lineno, line in enumerate(f.readlines(), start=1):
            line = line.strip().lower()
            if line != "" and line[:-6].strip() in tags.keys():
                tags[line[:-6].strip()].append(lineno)
            elif line != "" and line[:-4].strip() in tags.keys():
                tags[line[:-4].strip()].append(lineno)


def write_src_files(script_path: Path, workspace_dir: Path, tags: dict):
    for value in tags:
        filepath = workspace_dir.joinpath(value[0]).resolve()
        if str(filepath).endswith("yaml"):
            continue
        line_ranges = value[1:]
        lines_to_read = []
        # Open the file and read lines
        with open(script_path, 'r') as file:
            current_range = None  # To keep track of the current range being processed

            for i, line in enumerate(file):
                if current_range is None:
                    # Check if we are within a range
                    if line_ranges and i == line_ranges[0]:
                        current_range = (line_ranges.pop(0), line_ranges.pop(0))

                if current_range is not None:
                    lines_to_read.append(line)

                    # Check if we've reached the end of the current range
                    if i == current_range[1]:
                        current_range = None

        with open(filepath, "a") as f:
            for line in lines_to_read:
                f.write(line)


def write_data_yaml(script_path: Path, workspace_dir: Path, tags: dict):
    # Read code mentioned in between "# n collaborators starts/ends".
    # write collaborator names with settings as empty dictionary, and template as None.
    filepath, line_ranges = workspace_dir.joinpath(tags[0]).resolve(), tags[1:]
    with open(script_path, "r") as f:
        last_line = None
        data = f.readlines()
        for i in range(0, len(line_ranges), 2):
            start, end = line_ranges[i], line_ranges[i + 1]
            for line in data[start:end]:
                # Ignore commented lines
                if line.strip() != "" and not line.strip().startswith("#"):
                    # execute it to get list of collaborator names.
                    exec(line)
                    last_line = line

    # last line of the code (ignoring the comments) in between "# n collaborators starts/ends",
    # should be list of collaborators
    list_of_collaborator_names = vars()[last_line.split('=')[0].strip()]

    # Building dictionary for data.yaml
    data = {}
    for collab_name in list_of_collaborator_names:
        data[collab_name] = {
            "callable_func": {
                "settings": {}, "template": None
            }
        }
    data["aggregator"] = {
        "callable_func": {
            "settings": {}, "template": None
        }
    }
    with open(script_path, "r") as f:
        # Parse the code snippet using ast
        tree = ast.parse(f.read())

        # Extract the keyword arguments and their values passed to the
        # Collaborator/Aggregator constructor
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and (node.func.id == "Collaborator" or node.func.id == "Aggregator")
            ):
                for kw in node.keywords:
                    argument = kw.arg
                    value = None

                    if argument == "name" or argument == "num_cpus" or argument == "num_gpus":
                        print(f"\tIgnoring argument {argument}...")
                        continue
                    # Extract the value
                    if isinstance(kw.value, ast.NameConstant):  # Boolean
                        value = kw.value.value
                    elif isinstance(kw.value, ast.Num):  # Integer
                        value = kw.value.n
                    elif isinstance(kw.value, ast.Str):  # String
                        value = kw.value.s
                    else:
                        if value is None and node.func.id == "Collaborator":
                            # TODO
                            value = "src.collaborator_private_attrs.<name of the variable>"
                        elif value is None and node.func.id == "Aggregator":
                            # TODO
                            value = "src.aggregator_private_attrs.<name of the variable>"

                    for collab_name in list_of_collaborator_names:
                        if node.func.id == "Collaborator":
                            if argument == "private_attributes_callable":
                                data[collab_name]["callable_func"]["template"] = value
                            else:
                                data[collab_name]["callable_func"]["settings"][argument] = value
                        elif node.func.id == "Aggregator":
                            if argument == "private_attributes_callable":
                                data["aggregator"]["callable_func"]["template"] = value
                            else:
                                data["aggregator"]["callable_func"]["settings"][argument] = value
    with open(filepath, "w") as f:
        yaml.safe_dump(data, f)


def main(script_path: Path, workspace_output_dir: Path, template_workspace_dir: Path):
    tags = {
        "# collaborator private attributes": ["src/collaborator_private_attrs.py", ],
        "# aggregator private attributes": ["src/aggregator_private_attrs.py", ],
        "# pip requirements": ["requirements.txt", ],
        "# n collaborators": ["plan/data.yaml", ],
        "# flow class": ["src/flow.py", ],
        "# flspec object": ["plan/plan.yaml", ],
    }

    workspace_output_dir = workspace_output_dir.joinpath(f"{script_path.name}").resolve()
    print(f"Copying workspace template to {workspace_output_dir}...")
    copytree(str(template_workspace_dir), str(workspace_output_dir))

    print("Converting notebook to python script...")
    python_script_path = convert_to_python(script_path, workspace_output_dir)

    print("Build tags library...")
    build_tags_lib(python_script_path, tags)

    print("Writing python scripts for workspace...")
    write_src_files(python_script_path, workspace_output_dir, list(tags.values()))

    print("Writing data.yaml...")
    write_data_yaml(python_script_path, workspace_output_dir, tags["# n collaborators"])

    print("# TODO: Writing plan.yaml...")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-l", required=True, type=str, help="Absolute path of jupyter notebook"
                        + " script")
    parser.add_argument("-o", required=True, type=str, help="Output directory for generated"
                        + " workspace")
    parser.add_argument("-t", required=True, type=str, help="Absolute path of template workspace")

    parsed_args = parser.parse_args()

    notebook_path = Path(parsed_args.l).resolve()
    workspace_output_directory = Path(parsed_args.o).resolve()
    template_workspace_directory = Path(parsed_args.t).resolve()

    if not notebook_path.exists() and not notebook_path.is_file():
        raise FileNotFoundError(f"Jupyter notebook not found: {str(notebook_path)}")

    workspace_output_directory.mkdir(parents=True, exist_ok=True)

    main(notebook_path, workspace_output_directory, template_workspace_directory)
