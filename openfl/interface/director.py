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

from openfl.component.director import serve
from openfl.interface.cli_helper import WORKSPACE

logger = logging.getLogger(__name__)


@group()
@pass_context
def director(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'director'


@director.command(name='start')
@option('-c', '--director-config-path', default='director.yaml',
        help='The director config file path', type=ClickPath(exists=True))
@option('--disable-tls', default=False)
@option('-rc', '--root-cert-path', 'root_ca', default=None,
        help='Path to a root CA cert')
@option('-pk', '--private-key-path', 'key', default=None,
        help='Path to a private key')
@option('-oc', '--public-cert-path', 'cert', default=None,
        help='Path to a signed certificate')
def start(director_config_path, disable_tls, root_ca, key, cert):
    """Start the director service."""
    logger.info('🧿 Starting the Director Service.')
    with open(director_config_path) as stream:
        director_config = safe_load(stream)
    settings = director_config.get('settings', {})
    sample_shape = settings.get('sample_shape', '')
    target_shape = settings.get('target_shape', '')
    logger.info(f'Sample shape: {sample_shape}, target shape: {target_shape}')
    listen_ip = settings.get('listen_ip')
    listen_port = settings.get('listen_port')
    kwargs = {}
    if listen_ip:
        kwargs['listen_ip'] = listen_ip
    if listen_port:
        kwargs['listen_port'] = listen_port
    asyncio.run(serve(
        disable_tls=disable_tls,
        sample_shape=sample_shape,
        target_shape=target_shape,
        root_ca=root_ca, key=key, cert=cert, **kwargs
    ))


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
