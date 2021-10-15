# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl path checks."""

import os
from pathlib import Path
from typing import Union


def is_directory_traversal(directory: Union[str, Path]) -> bool:
    """Check for directory traversal."""
    cwd = os.path.abspath(os.getcwd())
    requested_path = os.path.relpath(directory, start=cwd)
    requested_path = os.path.abspath(requested_path)
    common_prefix = os.path.commonprefix([requested_path, cwd])
    return common_prefix != cwd
