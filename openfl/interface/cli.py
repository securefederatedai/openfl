#!/usr/bin/env python
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLI module."""

from click import Group, command, argument, group, clear
from click import echo, option, pass_context, style
from sys import argv
from pathlib import Path


def setup_logging(level='info'):
    """Initialize logging settings."""
    from logging import basicConfig, NOTSET, DEBUG, INFO
    from logging import WARNING, ERROR, CRITICAL
    from rich.console import Console
    from rich.logging import RichHandler
    import os

    console = Console(width=160)

    levels = \
        {
            'notset': NOTSET,
            'debug': DEBUG,
            'info': INFO,
            'warning': WARNING,
            'error': ERROR,
            'critical': CRITICAL
        }

    if level.lower() in ['debug', 'error']:
        os.environ['GRPC_VERBOSITY'] = level.upper()

    level = levels.get(level.lower(), levels['notset'])

    basicConfig(level=level, format='%(message)s',
                datefmt='[%X]', handlers=[RichHandler(console=console)])


def disable_warnings():
    """Disables CUDA warnings."""
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        uses = [
            f'{ctx.command_path}',
            '[options]',
            style('[command]', fg='blue'),
            style('[subcommand]', fg='cyan'),
            '[args]'
        ]

        formatter.write(style(
            'CORRECT USAGE\n\n', bold=True, fg='bright_black'))
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
            help = cmd.get_short_help_str()
            if level == 0:
                formatter.write(
                    f'\n{style(name, fg="blue", bold=True):<30}'
                    f' {style(help, bold=True)}' + '\n')
                formatter.write('─' * 80 + '\n')
            if level == 1:
                formatter.write(
                    f'  {style("⮞", fg="green")}'
                    f' {style(name, fg="cyan"):<21} {help}' + '\n')


@group(cls=CLI)
@option('-l', '--log-level', default='info', help='Logging verbosity level.')
@pass_context
def cli(context, log_level):
    """Command-line Interface."""
    context.ensure_object(dict)

    context.obj['log_level'] = log_level
    context.obj['fail'] = False
    context.obj['script'] = argv[0]
    context.obj['arguments'] = argv[1:]

    setup_logging(log_level)


@cli.resultcallback()
@pass_context
def end(context, result, **kwargs):
    """Print the result of the operation."""
    if context.obj['fail']:
        echo('\n ❌ :(')
    else:
        echo('\n ✔️ OK')


@command()
@pass_context
@argument('subcommand', required=False)
def help(context, subcommand):
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


def entry():
    """Entry point of the Command-Line Interface."""
    from importlib import import_module
    from sys import path

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(str(root))
    path.insert(0, str(work))

    clear()
    banner = 'Intel OpenFL - Secure Federated Learning at the Edge™'
    echo(style(f'{banner:<80}', bold=True, bg='bright_blue'))
    echo()

    # Setup logging immediately to suppress unnecessary warnings on import
    # This will be overridden later with user selected debugging level
    disable_warnings()

    for module in root.glob('*.py'):  # load command modules

        package = module.parent
        module = module.name.split('.')[0]

        if module.count('__init__') or module.count('cli'):
            continue

        command_group = import_module(module, package)

        cli.add_command(command_group.__getattribute__(module))

    try:
        cli()
    except Exception as e:
        error_handler(e)


if __name__ == '__main__':
    entry()
