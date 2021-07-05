# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""PKI CLI."""

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
from openfl.component.ca.ca import certify 
from openfl.component.ca.ca import remove_ca 
from openfl.component.ca.ca import get_token 
from openfl.component.ca.ca import start_ca


logger = logging.getLogger(__name__)

CA_URL = 'localhost:9123'


@group()
@pass_context
def pki(context):
    """Manage Step-ca PKI."""
    context.obj['group'] = 'pki'


@pki.command(name='start-ca')
@option('-p', '--ca-path', required=True,
        help='The ca path', type=ClickPath())
@option('--password', required=True)
@option('--ca-url', required=False, default=CA_URL)
def start_ca_(ca_path, password, ca_url):
    """Create a ca workspace."""
    start_ca(ca_path, ca_url, password)


@pki.command(name='remove-ca')
@option('-p', '--ca-path', required=True,
        help='The CA path', type=ClickPath())
def remove_ca_(ca_path):
    """Remove step-CA."""
    remove_ca(ca_path)


@pki.command(name='get-token')
@option('-n', '--name', required=True)
@option('--ca-url', required=False, default=CA_URL)
@option('-p', '--ca-path', default='.',
        help='The CA path', type=ClickPath())
def get_token_(name, ca_url, ca_path):
    """
    Create authentication token.

    Args:
        name: common name for following certificate
                    (aggregator fqdn or collaborator name)
        ca_url: full url of CA server
    """
    token = get_token(name, ca_url, ca_path)
    print('Token:')
    print(token)


@pki.command(name='certify')
@option('-n', '--name', required=True)
@option('-t', '--token', 'token_with_cert', required=True)
@option('-p', '--certs-path', required=False, default=Path('.') / 'cert',
        help='The ca path', type=ClickPath())
def certify_(name, token_with_cert, certs_path):
    """Create a collaborator manager workspace."""
    certs_path.mkdir(parents=True, exist_ok=True)
    certify(name, certs_path, token_with_cert)
