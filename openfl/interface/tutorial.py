# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tutorial module."""

from logging import getLogger

from click import group
from click import IntRange
from click import option
from click import pass_context

from openfl.utilities import click_types

logger = getLogger(__name__)


@group()
@pass_context
def tutorial(context):
    """Manage Jupyter notebooks."""
    context.obj['group'] = 'tutorial'


@tutorial.command()
@option('-ip', '--ip', required=False, type=click_types.IP_ADDRESS,
        help='IP address the Jupyter Lab that should start')
@option('-port', '--port', required=False, type=IntRange(1, 65535),
        help='The port the Jupyter Lab server will listen on')
def start(ip, port):
    """Start the Jupyter Lab from the tutorials directory."""
    from os import environ
    from subprocess import check_call
    from sys import executable

    from openfl.interface.cli_helper import TUTORIALS

    if 'VIRTUAL_ENV' in environ:
        venv = environ['VIRTUAL_ENV'].split('/')[-1]
        check_call([
            executable, '-m', 'ipykernel', 'install',
            '--user', '--name', f'{venv}'
        ], shell=False)

    jupyter_command = ['jupyter', 'lab', '--notebook-dir', f'{TUTORIALS}']

    if ip is not None:
        jupyter_command += ['--ip', f'{ip}']
    if port is not None:
        jupyter_command += ['--port', f'{port}']

    check_call(jupyter_command)
