# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
