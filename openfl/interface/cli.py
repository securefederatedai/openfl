#!/usr/bin/env python
# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLI module."""

import os
import re
import sys
import time
import warnings
from importlib import import_module
from pathlib import Path
from sys import argv, path

from click import (
    Group,
    argument,
    command,
    confirm,
    echo,
    group,
    open_file,
    option,
    pass_context,
    style,
)

import openfl
from openfl.utilities.logging import setup_logger


def disable_warnings():
    """Disables warnings."""

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)


class CLI(Group):
    """CLI class."""

    def __init__(self, name=None, commands=None, **kwargs):
        """
        Initialize CLI object.

        Args:
            name (str, optional): Name of the CLI group. Defaults to None.
            commands (dict, optional): Commands for the CLI group. Defaults
                to None.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(name, commands, **kwargs)
        self.commands = commands or {}

    def list_commands(self, ctx):
        """Display all available commands.

        Args:
            ctx (click.core.Context): Click context.

        Returns:
            dict: Available commands.
        """
        return self.commands

    def format_help(self, ctx, formatter):
        """Display user-friendly help.

        Args:
            ctx (click.core.Context): Click context.
            formatter (click.formatting.HelpFormatter): Click help formatter.
        """
        show_header()
        uses = [
            f"{ctx.command_path}",
            "[options]",
            style("[command]", fg="blue"),
            style("[subcommand]", fg="cyan"),
            "[args]",
        ]

        formatter.write(style("BASH COMPLETE ACTIVATION\n\n", bold=True, fg="bright_black"))
        formatter.write(
            "Run in terminal:\n"
            "   _FX_COMPLETE=bash_source fx > ~/.fx-autocomplete.sh\n"
            "   source ~/.fx-autocomplete.sh\n"
            "If ~/.fx-autocomplete.sh has already exist:\n"
            "   source ~/.fx-autocomplete.sh\n\n"
        )

        formatter.write(style("CORRECT USAGE\n\n", bold=True, fg="bright_black"))
        formatter.write(" ".join(uses) + "\n")

        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)

        formatter.write(style("\nGLOBAL OPTIONS\n\n", bold=True, fg="bright_black"))
        formatter.write_dl(opts)

        cmds = []
        for cmd in self.list_commands(ctx):
            cmd = self.get_command(ctx, cmd)
            cmds.append((cmd.name, cmd, 0))

            for sub in cmd.list_commands(ctx):
                sub = cmd.get_command(ctx, sub)
                cmds.append((sub.name, sub, 1))

        formatter.write(style("\nAVAILABLE COMMANDS\n", bold=True, fg="bright_black"))

        for name, cmd, level in cmds:
            help_str = cmd.get_short_help_str()
            if level == 0:
                formatter.write(
                    f"\n{style(name, fg='blue', bold=True):<30} {style(help_str, bold=True)}" + "\n"
                )
                formatter.write("─" * 80 + "\n")
            if level == 1:
                formatter.write(
                    f"  {style('*', fg='green')} {style(name, fg='cyan'):<21} {help_str}" + "\n"
                )

    def invoke(self, ctx):
        if ctx.params.get("version"):
            echo(f"OpenFL version: {openfl.__version__}")
            ctx.exit()
        super().invoke(ctx)


@group(cls=CLI)
@option("-l", "--log-level", default="info", help="Logging verbosity level.")
@option("--no-warnings", is_flag=True, help="Disable third-party warnings.")
@option("-v", "--version", is_flag=True, help="Show version")
@pass_context
def cli(context, log_level, no_warnings, version):
    """Command-line Interface."""

    context.ensure_object(dict)
    context.obj["log_level"] = log_level
    context.obj["fail"] = False
    context.obj["script"] = argv[0]
    context.obj["arguments"] = argv[1:]

    if no_warnings:
        # Setup logging immediately to suppress unnecessary warnings on import
        # This will be overridden later with user selected debugging level
        disable_warnings()
    log_file = os.getenv("LOG_FILE")
    # Validate log_file with tighter restrictions
    if log_file:
        log_file = os.path.normpath(log_file)
        if (
            not re.match(r"^logs/[\w\-.]+$", log_file)
            or ".." in log_file
            or log_file.startswith("/")
        ):
            raise ValueError("Invalid log file path")
        # Ensure the log file is in the 'logs' directory
        allowed_directory = Path("logs").resolve()
        full_path = (allowed_directory / log_file).resolve()
        if not str(full_path).startswith(str(allowed_directory)):
            raise ValueError("Log file path is not allowed")
    setup_logger(log_level, log_file)
    sys.stdout.reconfigure(encoding="utf-8")


@cli.result_callback()
@pass_context
def end(context, result, **kwargs):
    """
    Print the result of the operation.

    Args:
        context (click.core.Context): Click context.
        result: Result of the operation.
        **kwargs: Arbitrary keyword arguments.
    """
    if context.obj["fail"]:
        echo("\n ❌ :(")
    else:
        echo("\n ✔️ OK")


@command(name="help")
@pass_context
@argument("subcommand", required=False)
def help_(context, subcommand):
    """Display help.

    Args:
        context (click.core.Context): Click context.
        subcommand (str, optional): Subcommand to display help for. Defaults
            to None.
    """
    pass


def error_handler(error):
    """
    Handle the error.

    Args:
        error (Exception): Error to handle.
    """
    if "cannot import" in str(error):
        if "TensorFlow" in str(error):
            echo(
                style("EXCEPTION", fg="red", bold=True)
                + " : "
                + style(
                    "Tensorflow must be installed prior to running this command",
                    fg="red",
                )
            )
        if "PyTorch" in str(error):
            echo(
                style("EXCEPTION", fg="red", bold=True)
                + " : "
                + style(
                    "Torch must be installed prior to running this command",
                    fg="red",
                )
            )
    echo(style("EXCEPTION", fg="red", bold=True) + " : " + style(f"{error}", fg="red"))
    raise error


def review_plan_callback(file_name, file_path):
    """
    Review plan callback for Director and Envoy.

    Args:
        file_name (str): Name of the file to review.
        file_path (str): Path of the file to review.

    Returns:
        bool: True if the file is accepted, False otherwise.
    """
    echo(
        style(
            f"Please review the contents of {file_name} before proceeding...",
            fg="green",
            bold=True,
        )
    )
    # Wait for users to read the question before flashing the contents of the file.
    time.sleep(3)

    with open_file(file_path, "r") as f:
        echo(f.read())

    if confirm(style(f"Do you want to accept the {file_name}?", fg="green", bold=True)):
        echo(style(f"{file_name} accepted!", fg="green", bold=True))
        return True
    else:
        echo(style(f"EXCEPTION: {file_name} rejected!", fg="red", bold=True))
        return False


def show_header():
    """Show header."""

    banner = "OpenFL - Open Federated Learning"

    experimental = (
        Path(os.path.expanduser("~")).resolve().joinpath(".openfl", "experimental").resolve()
    )

    if os.path.exists(experimental):
        banner = "OpenFL - Open Federated Learning (Experimental)"

    echo(style(f"{banner:<80}", bold=True, bg="bright_blue"))
    echo()


def entry():
    """Entry point of the Command-Line Interface."""

    experimental = (
        Path(os.path.expanduser("~")).resolve().joinpath(".openfl", "experimental").resolve()
    )

    root = Path(__file__).parent.resolve()

    if experimental.exists():
        root = root.parent.joinpath("experimental", "workflow", "interface", "cli").resolve()

    work = Path.cwd().resolve()
    path.append(str(root))
    path.insert(0, str(work))

    for module in root.glob("*.py"):  # load command modules
        package = module.parent
        module = module.name.split(".")[0]

        if module.count("__init__") or module.count("cli"):
            continue

        command_group = import_module(module, package)

        cli.add_command(command_group.__getattribute__(module))

    try:
        cli(max_content_width=120)
    except Exception as e:
        error_handler(e)


if __name__ == "__main__":
    entry()
