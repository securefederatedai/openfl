# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Director CLI."""

import logging
import shutil
from pathlib import Path

import click
import sys
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from yaml import safe_load

from openfl.component.director import Director
from openfl.interface.cli_helper import WORKSPACE
from openfl.transport import DirectorGRPCServer

logger = logging.getLogger(__name__)


@group()
@pass_context
def director(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'director'


# @director.command(name='start')
# @option('-c', '--director-config-path', default='director.yaml',
#         help='The director config file path', type=ClickPath(exists=True))
# @option('--tls/--disable-tls', default=True,
#         is_flag=True, help='Use TLS or not (By default TLS is enabled)')
# @option('-rc', '--root-cert-path', 'root_certificate', default=None,
#         help='Path to a root CA cert')
# @option('-pk', '--private-key-path', 'private_key', default=None,
#         help='Path to a private key')
# @option('-oc', '--public-cert-path', 'certificate', default=None,
#         help='Path to a signed certificate')
def start(director_config_path, tls, root_certificate, private_key, certificate):
    """Start the director service."""
    logger.info('🧿 Starting the Director Service.')
    with open(director_config_path) as stream:
        director_config = safe_load(stream)
    settings = director_config.get('settings', {})
    sample_shape = settings.get('sample_shape', '')
    target_shape = settings.get('target_shape', '')
    logger.info(f'Sample shape: {sample_shape}, target shape: {target_shape}')
    listen_host = settings.get('listen_host')
    listen_port = settings.get('listen_port')
    root_certificate = root_certificate or settings.get('root_certificate')
    private_key = private_key or settings.get('private_key')
    certificate = certificate or settings.get('certificate')
    kwargs = {}
    if listen_host:
        kwargs['listen_host'] = listen_host
    if listen_port:
        kwargs['listen_port'] = listen_port
    director_server = DirectorGRPCServer(
        director_cls=Director,
        tls=tls,
        sample_shape=sample_shape,
        target_shape=target_shape,
        root_certificate=root_certificate,
        private_key=private_key,
        certificate=certificate,
        settings=settings,
        **kwargs
    )
    director_server.start()


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


if __name__ == '__main__':
    start(
        director_config_path='/home/dmitry/code/openfl/openfl-tutorials/interactive_api/Director_Pytorch_Kvasir_UNET/director_folder/director_config.yaml',
        tls=False,
        root_certificate=None,
        private_key=None,
        certificate=None
    )
