# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Experimental CLI."""

import os
from pathlib import Path

from click import group
from click import pass_context


@group()
@pass_context
def experimental(context):
    """Manage Experimental Environment."""
    context.obj["group"] = "experimental"


@experimental.command(name="activate")
def activate():
    """Activate experimental environment."""
    settings = Path("~").expanduser().joinpath(
        ".openfl", "experimental").resolve()

    with open(settings, "w") as f:
        f.write("experimental")
