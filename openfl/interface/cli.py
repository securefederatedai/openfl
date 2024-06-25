#!/usr/bin/env python
# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLI module."""

import os

from click import argument
from click import command
from click import confirm
from click import echo
from click import Group
from click import group
from click import open_file
from click import option
from click import pass_context
from click import style
import time
import sys
from openfl.utilities import add_log_level


def setup_logging(level='info', log_file=None):
    """Initialize logging settings."""
    import logging
    from logging import basicConfig

    from rich.console import Console
    from rich.logging import RichHandler

    metric = 25
    add_log_level('METRIC', metric)

    if isinstance(level, str):
        level = level.upper()

    handlers = []
    if log_file:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s %(filename)s:%(lineno)d'
        )
        fh.setFormatter(formatter)
        handlers.append(fh)

    console = Console(width=160)
    handlers.append(RichHandler(console=console))
    basicConfig(level=level, format='%(message)s',
                datefmt='[%X]', handlers=handlers)


def disable_warnings():
    """Disables warnings."""
    import os
    import warnings

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)


class CLI(Group):
    """CLI class."""

    def __init__(self, name=None, commands=None, **kwargs):
        """Initialize."""
        super(CLI, self).__init__(name, commands, **kwargs)
        self.commands = commands or {}

    def list_commands(self, ctx):
        """Display all available commands."""
        return self.commands

    def format_help(self, ctx, formatter):
        """Dislpay user-friendly help."""
        show_header()
        uses = [
            f'{ctx.command_path}',
            '[options]',
            style('[command]', fg='blue'),
            style('[subcommand]', fg='cyan'),
            '[args]'
        ]

        formatter.write(style('BASH COMPLETE ACTIVATION\n\n', bold=True, fg='bright_black'))
        formatter.write(
            'Run in terminal:\n'
            '   _FX_COMPLETE=bash_source fx > ~/.fx-autocomplete.sh\n'
            '   source ~/.fx-autocomplete.sh\n'
            'If ~/.fx-autocomplete.sh has already exist:\n'
            '   source ~/.fx-autocomplete.sh\n\n'
        )

        formatter.write(style('CORRECT USAGE\n\n', bold=True, fg='bright_black'))
        formatter.write(' '.join(uses) + '\n')

        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)

        formatter.write(style(
            '\nGLOBAL OPTIONS\n\n', bold=True, fg='bright_black'))
        formatter.write_dl(opts)

        cmds = []
        for cmd in self.list_commands(ctx):
            cmd = self.get_command(ctx, cmd)
            cmds.append((cmd.name, cmd, 0))

            for sub in cmd.list_commands(ctx):
                sub = cmd.get_command(ctx, sub)
                cmds.append((sub.name, sub, 1))

        formatter.write(style(
            '\nAVAILABLE COMMANDS\n', bold=True, fg='bright_black'))

        for name, cmd, level in cmds:
            help_str = cmd.get_short_help_str()
            if level == 0:
                formatter.write(
                    f'\n{style(name, fg="blue", bold=True):<30}'
                    f' {style(help_str, bold=True)}' + '\n')
                formatter.write('─' * 80 + '\n')
            if level == 1:
                formatter.write(
                    f'  {style("*", fg="green")}'
                    f' {style(name, fg="cyan"):<21} {help_str}' + '\n')


@group(cls=CLI)
@option('-l', '--log-level', default='info', help='Logging verbosity level.')
@option('--no-warnings', is_flag=True, help='Disable third-party warnings.')
@pass_context
def cli(context, log_level, no_warnings):
    """Command-line Interface."""
    import os
    from sys import argv

    context.ensure_object(dict)
    context.obj['log_level'] = log_level
    context.obj['fail'] = False
    context.obj['script'] = argv[0]
    context.obj['arguments'] = argv[1:]

    if no_warnings:
        # Setup logging immediately to suppress unnecessary warnings on import
        # This will be overridden later with user selected debugging level
        disable_warnings()
    log_file = os.getenv('LOG_FILE')
    setup_logging(log_level, log_file)
    sys.stdout.reconfigure(encoding='utf-8')


@cli.result_callback()
@pass_context
def end(context, result, **kwargs):
    """Print the result of the operation."""
    if context.obj['fail']:
        echo('\n ❌ :(')
    else:
        echo('\n ✔️ OK')


@command(name='help')
@pass_context
@argument('subcommand', required=False)
def help_(context, subcommand):
    """Display help."""
    pass


def error_handler(error):
    """Handle the error."""
    if 'cannot import' in str(error):
        if 'TensorFlow' in str(error):
            echo(style('EXCEPTION', fg='red', bold=True) + ' : ' + style(
                'Tensorflow must be installed prior to running this command',
                fg='red'))
        if 'PyTorch' in str(error):
            echo(style('EXCEPTION', fg='red', bold=True) + ' : ' + style(
                'Torch must be installed prior to running this command',
                fg='red'))
    echo(style('EXCEPTION', fg='red', bold=True)
         + ' : ' + style(f'{error}', fg='red'))
    raise error


def review_plan_callback(file_name, file_path):
    """Review plan callback for Director and Envoy."""
    echo(style(
        f'Please review the contents of {file_name} before proceeding...',
        fg='green',
        bold=True))
    # Wait for users to read the question before flashing the contents of the file.
    time.sleep(3)

    with open_file(file_path, 'r') as f:
        echo(f.read())

    if confirm(style(f'Do you want to accept the {file_name}?', fg='green', bold=True)):
        echo(style(f'{file_name} accepted!', fg='green', bold=True))
        return True
    else:
        echo(style(f'EXCEPTION: {file_name} rejected!', fg='red', bold=True))
        return False


def show_header():
    """Show header."""
    from pathlib import Path

    banner = 'OpenFL - Open Federated Learning'

    experimental = Path(os.path.expanduser("~")).resolve().joinpath(
        ".openfl", "experimental").resolve()

    if os.path.exists(experimental):
        banner = 'OpenFL - Open Federated Learning (Experimental)'

    echo(style(f'{banner:<80}', bold=True, bg='bright_blue'))
    echo()


def entry():
    """Entry point of the Command-Line Interface."""
    from importlib import import_module
    from pathlib import Path
    from sys import path

    experimental = Path(os.path.expanduser("~")).resolve().joinpath(
        ".openfl", "experimental").resolve()

    root = Path(__file__).parent.resolve()

    if experimental.exists():
        root = root.parent.joinpath("experimental", "interface", "cli").resolve()

    work = Path.cwd().resolve()
    path.append(str(root))
    path.insert(0, str(work))

    for module in root.glob('*.py'):  # load command modules

        package = module.parent
        module = module.name.split('.')[0]

        if module.count('__init__') or module.count('cli'):
            continue

        command_group = import_module(module, package)

        cli.add_command(command_group.__getattribute__(module))

    try:
        cli(max_content_width=120)
    except Exception as e:
        error_handler(e)


if __name__ == '__main__':
    entry()
