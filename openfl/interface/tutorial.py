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

"""Tutorial module."""
from logging import getLogger
from os import environ, sep
from subprocess import check_call  # nosec
from sys import executable

from click import IntRange, group, option, pass_context

from openfl.interface.cli_helper import TUTORIALS
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


    if 'VIRTUAL_ENV' in environ:
        venv = environ['VIRTUAL_ENV'].split(sep)[-1]
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
