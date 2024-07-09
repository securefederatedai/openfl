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

import os
from pathlib import Path

from click import group, pass_context


@group()
@pass_context
def experimental(context):
    """Manage Experimental Environment."""
    context.obj["group"] = "experimental"


@experimental.command(name="deactivate")
def deactivate():
    """Deactivate experimental environment."""
    settings = (
        Path("~").expanduser().joinpath(".openfl", "experimental").resolve()
    )

    os.remove(settings)
