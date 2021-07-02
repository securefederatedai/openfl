# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Director CLI."""

import logging
import shutil
import os
from pathlib import Path

import click
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from yaml import safe_load

from openfl.interface.cli_helper import WORKSPACE
from openfl.component.ca.ca import get_token 
from openfl.component.ca.ca import certify 
from openfl.component.ca.ca import download_step_bin
from openfl.component.ca.ca import get_bin_names
from openfl.component.ca.ca import run
from openfl.component.ca.ca import create


logger = logging.getLogger(__name__)


@group()
@pass_context
def api_layer(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'api-layer'

@api_layer.command(name='certify')
@option('-t', '--token','token_with_cert', required=True)
def certify_(token_with_cert):
    """Create a collaborator manager workspace."""
    if not os.path.exists('./cert'):
        prefix = Path('.')
        (prefix / 'cert').mkdir(parents=True, exist_ok=True)
    certify('api', '','./cert', token_with_cert)