# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl path checks."""

import os
from pathlib import Path
from typing import Union


def is_directory_traversal(directory: Union[str, Path]) -> bool:
    """Check for directory traversal.

    This function checks if the provided directory is a subdirectory of the
    current working directory.
    It returns `True` if the directory is not a subdirectory (i.e., it is a
    directory traversal), and `False` otherwise.

    Args:
        directory (Union[str, Path]): The directory to check.

    Returns:
        bool: `True` if the directory is a directory traversal, `False`
            otherwise.
    """
    cwd = os.path.abspath(os.getcwd())
    requested_path = os.path.relpath(directory, start=cwd)
    requested_path = os.path.abspath(requested_path)
    common_prefix = os.path.commonprefix([requested_path, cwd])
    return common_prefix != cwd
