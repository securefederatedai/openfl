# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Module with auxiliary CLI helper functions."""

import os
import re
from itertools import islice
from os import environ
from pathlib import Path
from sys import argv

from click import echo, style
from yaml import FullLoader, load

FX = argv[0]

SITEPACKS = Path(__file__).parent.parent.parent.parent.parent.parent
WORKSPACE = SITEPACKS / "openfl-workspace" / "experimental" / "workflow" / "FederatedRuntime"
TUTORIALS = SITEPACKS / "openfl-tutorials"
OPENFL_USERDIR = Path.home() / ".openfl"
CERT_DIR = Path("cert").absolute()


def pretty(o):
    """Pretty-print the dictionary given."""
    m = max(map(len, o.keys()))

    for k, v in o.items():
        echo(style(f"{k:<{m}} : ", fg="blue") + style(f"{v}", fg="cyan"))


def print_tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
):
    """Given a directory Path object print a visual tree structure."""
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "

    echo("\nNew experimental workspace directory structure:")

    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 0

    def inner(dir_path: Path, prefix: str = "", level=-1):
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    echo(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        echo(line)
    if next(iterator, None):
        echo(f"... length_limit, {length_limit}, reached, counted:")
    echo(f"\n{directories} directories" + (f", {files} files" if files else ""))


def get_workspace_parameter(name):
    """Get a parameter from the workspace config file (.workspace)."""
    # Update the .workspace file to show the current workspace plan
    workspace_file = ".workspace"

    with open(workspace_file, "r", encoding="utf-8") as f:
        doc = load(f, Loader=FullLoader)

    if not doc:  # YAML is not correctly formatted
        doc = {}  # Create empty dictionary

    if name not in doc.keys() or not doc[name]:  # List doesn't exist
        return ""
    else:
        return doc[name]


def check_varenv(env: str = "", args: dict = None):
    """Update "args" (dictionary) with <env: env_value> if env has a defined
    value in the host."""
    if args is None:
        args = {}
    env_val = environ.get(env)
    if env and (env_val is not None):
        args[env] = env_val

    return args


def get_fx_path(curr_path=""):
    """Return the absolute path to fx binary."""

    match = re.search("lib", curr_path)
    idx = match.end()
    path_prefix = curr_path[0:idx]
    bin_path = re.sub("lib", "bin", path_prefix)
    fx_path = os.path.join(bin_path, "fx")

    return fx_path


def remove_line_from_file(pkg, filename):
    """Remove line that contains `pkg` from the `filename` file."""
    with open(filename, "r+", encoding="utf-8") as f:
        d = f.readlines()
        f.seek(0)
        for i in d:
            if pkg not in i:
                f.write(i)
        f.truncate()


def replace_line_in_file(line, line_num_to_replace, filename):
    """Replace line at `line_num_to_replace` with `line`."""
    with open(filename, "r+", encoding="utf-8") as f:
        d = f.readlines()
        f.seek(0)
        for idx, i in enumerate(d):
            if idx == line_num_to_replace:
                f.write(line)
            else:
                f.write(i)
        f.truncate()
