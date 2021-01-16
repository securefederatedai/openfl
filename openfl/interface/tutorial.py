# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tutorial module."""

from sys import executable
from subprocess import check_call
from os import environ
from logging import getLogger

from click import group, option, pass_context

from openfl.interface.cli_helper import TUTORIALS

logger = getLogger(__name__)


@group()
@pass_context
def tutorial(context):
    """Manage Jupyter notebooks."""
    context.obj['group'] = 'tutorial'


@tutorial.command()
@pass_context
@option('-ip', '--ip', required=False,
        help='IP address the notebook that should start')
def start(context, ip):
    """Start the Jupyter notebook from the tutorials directory."""
    if 'VIRTUAL_ENV' in environ:
        venv = environ['VIRTUAL_ENV'].split('/')[-1]
        check_call([
            executable, '-m', 'ipykernel', 'install',
            '--user', '--name', f'{venv}'
        ], shell=False)

    jupyter_command = ['jupyter', 'notebook', '--notebook-dir', f'{TUTORIALS}']

    if ip is not None:
        jupyter_command += ['--ip', f'{ip}']

    check_call(jupyter_command)
