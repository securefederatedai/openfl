# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tutorial module."""

from logging import getLogger

from click import group
from click import IntRange
from click import option
from click import pass_context


from openfl.utilities.utils import IPADRESS

logger = getLogger(__name__)


@group()
@pass_context
def tutorial(context):
    """Manage Jupyter notebooks."""
    context.obj['group'] = 'tutorial'


@tutorial.command()
@pass_context
@option('-ip', '--ip', required=False, type=IPADRESS,
        help='IP address the notebook that should start')
@option('-port', '--port', required=False, type=IntRange(1, 65535),
        help='The port the notebook server will listen on')
def start(context, ip, port):
    """Start the Jupyter notebook from the tutorials directory."""
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

    jupyter_command = ['jupyter', 'notebook', '--notebook-dir', f'{TUTORIALS}']

    if ip is not None:
        jupyter_command += ['--ip', f'{ip}']
    if port is not None:
        jupyter_command += ['--port', f'{port}']

    check_call(jupyter_command)
