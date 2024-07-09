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

"""Experimental CLI."""
from logging import getLogger
from pathlib import Path
from subprocess import check_call
from sys import executable

from click import group, pass_context

import openfl

logger = getLogger(__name__)


@group()
@pass_context
def experimental(context):
    """Manage Experimental Environment."""
    context.obj["group"] = "experimental"


@experimental.command(name="activate")
def activate():
    """Activate experimental environment."""
    settings = Path("~").expanduser().joinpath(
        ".openfl").resolve()
    settings.mkdir(parents=False, exist_ok=True)
    settings = settings.joinpath("experimental").resolve()



    rf = Path(openfl.__file__).parent.parent.resolve().joinpath(
        "openfl-tutorials", "experimental", "requirements_workflow_interface.txt").resolve()

    if rf.is_file():
        check_call(
            [executable, '-m', 'pip', 'install', '-r', rf],
            shell=False
        )
    else:
        logger.warning(f"Requirements file {rf} not found.")

    with open(settings, "w") as f:
        f.write("experimental")
