# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
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
            line = line.strip()
            if line != "" and line in tags.keys():
                tags[line].append(lineno)
            elif line != "" and line[:-3].strip() in tags.keys():
                tags[line[:-3].strip()].append(lineno)


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
    import ast

    with open(script_path, "r") as f:
        # Parse the code snippet using ast
        tree = ast.parse(f.read())

        # Extract the keyword arguments and their values passed to the Collaborator constructor
        arguments_and_values = {
            "Collaborator": {}, "Aggregator": {}
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and (node.func.id == "Collaborator" or node.func.id == "Aggregator")
            ):
                for kw in node.keywords:
                    argument = kw.arg
                    value = None

                    # Extract the value using a custom function
                    if isinstance(kw.value, ast.NameConstant):  # Binary
                        value = kw.value.value
                    elif isinstance(kw.value, ast.Num):  # Integer
                        value = kw.value.n
                    elif isinstance(kw.value, ast.Str):  # String
                        value = kw.value.s

                    arguments_and_values[node.func.id][argument] = value
    print(arguments_and_values)


def main(script_path: Path, workspace_output_dir: Path, template_workspace_dir: Path):
    tags = {
        "# Collaborator private attributes": ["src/collaborator_private_attrs.py", ],
        "# Aggregator private attributes": ["src/aggregator_private_attrs.py", ],
        "# pip requirements": ["requirements.txt", ],
        "# n Collaborators": ["plan/data.yaml", ],
        "# Flow Class": ["src/flow.py", ],
        "# FLSpec object": ["plan/plan.yaml", ],
    }

    workspace_output_dir = workspace_output_dir.joinpath(f"{script_path.name}").resolve()
    print(f"Copying workspace template to {workspace_output_dir}...")
    copytree(str(template_workspace_dir), str(workspace_output_dir))

    print("Converting notebook to python script...")
    python_script_path = convert_to_python(script_path, workspace_output_dir)

    print("Build tag libraries...")
    build_tags_lib(python_script_path, tags)

    print("Finding objects...")
    write_src_files(python_script_path, workspace_output_dir, list(tags.values()))

    print("Writing data.yaml...")
    write_data_yaml(python_script_path, workspace_output_dir, tags["# n Collaborators"])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-l", required=True, type=str, help="Absolute path to jupyter notebook"
                        + " script")
    parser.add_argument("-o", required=True, type=str, help="Output directory for generated"
                        + " workspace")
    parser.add_argument("-t", required=True, type=str, help="Absolute path to template workspace")

    parsed_args = parser.parse_args()

    notebook_path = Path(parsed_args.l).resolve()
    workspace_output_directory = Path(parsed_args.o).resolve()
    template_workspace_directory = Path(parsed_args.t).resolve()

    if not notebook_path.exists() and not notebook_path.is_file():
        raise FileNotFoundError(f"Python script not found {str(notebook_path)}")

    workspace_output_directory.mkdir(parents=True, exist_ok=True)

    main(notebook_path, workspace_output_directory, template_workspace_directory)
