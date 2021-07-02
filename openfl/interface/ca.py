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
def ca(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'ca'



@ca.command(name='create')
@option('-p', '--ca-path', required=True,
        help='The ca path', type=ClickPath())
@option('--ca-url', required=True)
@option('--password', required=True)
def create_(ca_path, ca_url, password):
    """Create a ca workspace."""
    create(ca_path, ca_url, password)

@ca.command(name='certify')
@option('-f', '--fqdn', required=True,
        help='fqdn')
@option('-t', '--token', 'token_with_cert', required=True,
        help='token')
def certify_(fqdn, token_with_cert):
    """Create a collaborator manager workspace."""
    certify(fqdn, 'agg', token_with_cert)

@ca.command(name='get-token')
@option('-n', '--name', required=True)
@option('--ca-url', required=True)
def get_token_(name, ca_url):
    """
    Create authentication token.

    Args:
        name: common name for following certificate
                    (aggregator fqdn or collaborator name)
        ca_url: full url of CA server
    """
    token = get_token(name, ca_url)
    print('Token:')
    print(token)



@ca.command(name='run')
def run_():
    """
    Run CA server
    """
    run()