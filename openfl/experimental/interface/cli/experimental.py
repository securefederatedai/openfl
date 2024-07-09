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
