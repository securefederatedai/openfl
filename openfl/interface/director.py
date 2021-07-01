# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Director CLI."""

import asyncio
import logging
import shutil
import sys
from pathlib import Path

import click
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from yaml import safe_load

from openfl.component.director.director import serve
from openfl.interface.cli_helper import WORKSPACE
from openfl.component.ca.ca import get_token 
from openfl.component.ca.ca import certify 

logger = logging.getLogger(__name__)


@group()
@pass_context
def director(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'director'


@director.command(name='start')
@option('-c', '--director-config-path', default='director.yaml',
        help='The director config file path', type=ClickPath(exists=True))
def start(director_config_path):
    """Start the director service."""
    logger.info('🧿 Starting the Director Service.')
    with open(director_config_path) as stream:
        director_config = safe_load(stream)
    settings = director_config.get('settings', {})
    sample_shape = settings.get('sample_shape', '').split(',')
    target_shape = settings.get('target_shape', '').split(',')
    logger.info(f'Sample shape: {sample_shape}, target shape: {target_shape}')
    asyncio.run(serve(sample_shape=sample_shape, target_shape=target_shape))


@director.command(name='create-workspace')
@option('-p', '--director-path', required=True,
        help='The director path', type=ClickPath())
def create(director_path):
    """Create a director workspace."""
    director_path = Path(director_path)
    if director_path.exists():
        if not click.confirm('Director workspace already exists. Recreate?', default=True):
            sys.exit(1)
        shutil.rmtree(director_path)
    (director_path / 'cert').mkdir(parents=True, exist_ok=True)
    (director_path / 'logs').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORKSPACE / 'default/director.yaml', director_path / 'director.yaml')

@director.command(name='certify')
@option('-f', '--fqdn', required=True,
        help='fqdn')
@option('-t', '--token', 'token_with_cert', required=True,
        help='token')
def certify_(fqdn, token_with_cert):
    """Create a collaborator manager workspace."""
    certify(fqdn, 'agg', 'cert', token_with_cert)

# @director.command(name='get-token')
# @option('-n', '--name', required=True)
# @option('--ca-url', required=True)
# def get_token_(name, ca_url):
#     """
#     Create authentication token.

#     Args:
#         name: common name for following certificate
#                     (aggregator fqdn or collaborator name)
#         ca_url: full url of CA server
#     """
#     token = get_token(name, ca_url)
#     print('Token:')
#     print(token)
