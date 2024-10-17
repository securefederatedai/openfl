# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


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
    settings = Path("~").expanduser().joinpath(".openfl").resolve()
    settings.mkdir(parents=False, exist_ok=True)
    settings = settings.joinpath("experimental").resolve()

    rf = (
        Path(openfl.__file__)
        .parent.parent.resolve()
        .joinpath(
            "openfl-tutorials",
            "experimental",
            "workflow_interface_requirements.txt",
        )
        .resolve()
    )

    if rf.is_file():
        check_call([executable, "-m", "pip", "install", "-r", rf], shell=False)
    else:
        logger.warning(f"Requirements file {rf} not found.")

    with open(settings, "w") as f:
        f.write("experimental")
