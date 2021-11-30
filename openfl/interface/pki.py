# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""PKI CLI."""

import logging
import os
from pathlib import Path

from click import group
from click import option
from click import pass_context
from click import password_option
from click import Path as ClickPath

from openfl.component.ca.ca import CA_CONFIG_JSON
from openfl.component.ca.ca import CA_PASSWORD_FILE
from openfl.component.ca.ca import CA_PKI_DIR
from openfl.component.ca.ca import CA_STEP_CONFIG_DIR
from openfl.component.ca.ca import certify
from openfl.component.ca.ca import get_ca_bin_paths
from openfl.component.ca.ca import get_token
from openfl.component.ca.ca import install
from openfl.component.ca.ca import remove_ca
from openfl.component.ca.ca import run_ca

logger = logging.getLogger(__name__)

CA_URL = 'localhost:9123'


@group()
@pass_context
def pki(context):
    """Manage Step-ca PKI."""
    context.obj['group'] = 'pki'


@pki.command(name='run')
@option('-p', '--ca-path', required=True,
        help='The ca path', type=ClickPath())
def run(ca_path):
    """Run CA server."""
    ca_path = Path(ca_path).absolute()
    step_config_dir = ca_path / CA_STEP_CONFIG_DIR
    pki_dir = ca_path / CA_PKI_DIR
    password_file = pki_dir / CA_PASSWORD_FILE
    ca_json = step_config_dir / CA_CONFIG_JSON
    _, step_ca_path = get_ca_bin_paths(ca_path)
    if (not os.path.exists(step_config_dir) or not os.path.exists(pki_dir)
            or not os.path.exists(password_file) or not os.path.exists(ca_json)
            or not os.path.exists(step_ca_path)):
        logger.warning('CA is not installed or corrupted, please install it first')
        return
    run_ca(step_ca_path, password_file, ca_json)


@pki.command(name='install')
@option('-p', '--ca-path', required=True,
        help='The ca path', type=ClickPath())
@password_option(prompt='The password will encrypt some ca files \nEnter the password')
@option('--ca-url', required=False, default=CA_URL)
def install_(ca_path, password, ca_url):
    """Create a ca workspace."""
    ca_path = Path(ca_path).absolute()
    install(ca_path, ca_url, password)


@pki.command(name='uninstall')
@option('-p', '--ca-path', required=True,
        help='The CA path', type=ClickPath())
def uninstall(ca_path):
    """Remove step-CA."""
    ca_path = Path(ca_path).absolute()
    remove_ca(ca_path)


@pki.command(name='get-token')
@option('-n', '--name', required=True)
@option('--ca-url', required=False, default=CA_URL)
@option('-p', '--ca-path', default='.',
        help='The CA path', type=ClickPath(exists=True))
def get_token_(name, ca_url, ca_path):
    """
    Create authentication token.

    Args:
        name: common name for following certificate
                    (aggregator fqdn or collaborator name)
        ca_url: full url of CA server
        ca_path: the path to CA binaries
    """
    ca_path = Path(ca_path).absolute()
    token = get_token(name, ca_url, ca_path)
    print('Token:')
    print(token)


@pki.command(name='certify')
@option('-n', '--name', required=True)
@option('-t', '--token', 'token_with_cert', required=True)
@option('-c', '--certs-path', required=False, default=Path('.') / 'cert',
        help='The path where certificates will be stored', type=ClickPath())
@option('-p', '--ca-path', default='.', help='The path to CA client',
        type=ClickPath(exists=True), required=False)
def certify_(name, token_with_cert, certs_path, ca_path):
    """Create an envoy workspace."""
    certs_path = Path(certs_path).absolute()
    ca_path = Path(ca_path).absolute()
    certs_path.mkdir(parents=True, exist_ok=True)
    certify(name, certs_path, token_with_cert, ca_path)
