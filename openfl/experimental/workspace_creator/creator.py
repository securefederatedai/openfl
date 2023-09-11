# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from pathlib import Path
from shutil import copytree
from glob import glob


def convert_to_python(notebook_path: str, output_path: str):
    from nbdev.export import nb_export
    nb_export(notebook_path, output_path)

    return glob(os.path.join(output_path, "*.py"), recursive=False)[0]


def build_tags_lib(script_path: Path, tags: dict):
    tags_dict = deepcopy(tags)

    with open(script_path, "r") as f:
        for lineno, line in enumerate(f.readlines(), start=1):
            line = line.strip()
            if line != "" and line in tags.keys():
                tags_dict[line].append(lineno)
            elif line != "" and line[:-3].strip() in tags.keys():
                tags_dict[line[:-3].strip()].append(lineno)

    return tags_dict


def main(script_path: Path, workspace_output_dir: Path, template_workspace_dir: Path):
    # tags = ["pip requirements", "Collaborator private attributes",
    #         "Aggregator private attributes", "Flow Class", "n Collaborators",
    #         "FLSpec object"]
    tags = {
        "# Collaborator private attributes": ["collaborator_private_attrs.py", ],
        "# Aggregator private attributes": ["aggregator_private_attrs.py", ],
        "# pip requirements": ["requirements.txt", ],
        "# n Collaborators": ["data.yaml", ],
        "# Flow Class": ["flow.py", ],
        "# FLSpec object": ["plan.yaml", ],
    }

    python_script_path = workspace_output_dir.joinpath(f"{script_path.name}").resolve()
    print(f"Copying workspace template to {python_script_path}...")
    copytree(str(template_workspace_dir), str(python_script_path))

    print("Converting notebook to python script...")
    python_script_path = convert_to_python(script_path, python_script_path)

    print("Build tag libraries...")
    tags_dict = build_tags_lib(python_script_path, tags)
    print(tags_dict)


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
