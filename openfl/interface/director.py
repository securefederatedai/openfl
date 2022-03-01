# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Director CLI."""

import logging
import shutil
import sys
from pathlib import Path

import click
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from dynaconf import Validator

from openfl.component.director import Director
from openfl.interface.cli_helper import WORKSPACE
from openfl.transport import DirectorGRPCServer
from openfl.utilities import merge_configs
from openfl.utilities.path_check import is_directory_traversal

logger = logging.getLogger(__name__)


@group()
@pass_context
def director(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'director'


@director.command(name='start')
@option('-c', '--director-config-path', default='director.yaml',
        help='The director config file path', type=ClickPath(exists=True))
@option('--tls/--disable-tls', default=True,
        is_flag=True, help='Use TLS or not (By default TLS is enabled)')
@option('-rc', '--root-cert-path', 'root_certificate', required=False,
        type=ClickPath(exists=True), default=None,
        help='Path to a root CA cert')
@option('-pk', '--private-key-path', 'private_key', required=False,
        type=ClickPath(exists=True), default=None,
        help='Path to a private key')
@option('-oc', '--public-cert-path', 'certificate', required=False,
        type=ClickPath(exists=True), default=None,
        help='Path to a signed certificate')
def start(director_config_path, tls, root_certificate, private_key, certificate):
    """Start the director service."""
    director_config_path = Path(director_config_path).absolute()
    logger.info('🧿 Starting the Director Service.')
    if is_directory_traversal(director_config_path):
        click.echo('The director config file path is out of the openfl workspace scope.')
        sys.exit(1)
    config = merge_configs(
        settings_files=director_config_path,
        overwrite_dict={
            'root_certificate': root_certificate,
            'private_key': private_key,
            'certificate': certificate,
        },
        validators=[
            Validator('settings.listen_host', default='localhost'),
            Validator('settings.listen_port', default=50051, gte=1024, lte=65535),
            Validator('settings.sample_shape', default=[]),
            Validator('settings.target_shape', default=[]),
            Validator('settings.envoy_health_check_period', gte=1, lte=24 * 60 * 60),
        ],
        value_transform=[
            ('settings.sample_shape', lambda x: list(map(str, x))),
            ('settings.target_shape', lambda x: list(map(str, x))),
        ],
    )

    logger.info(
        f'Sample shape: {config.settings.sample_shape}, '
        f'target shape: {config.settings.target_shape}'
    )

    if config.root_certificate:
        config.root_certificate = Path(config.root_certificate).absolute()

    if config.private_key:
        config.private_key = Path(config.private_key).absolute()

    if config.certificate:
        config.certificate = Path(config.certificate).absolute()

    director_server = DirectorGRPCServer(
        director_cls=Director,
        tls=tls,
        sample_shape=config.settings.sample_shape,
        target_shape=config.settings.target_shape,
        root_certificate=config.root_certificate,
        private_key=config.private_key,
        certificate=config.certificate,
        settings=config.settings,
        listen_host=config.settings.listen_host,
        listen_port=config.settings.listen_port,
    )
    director_server.start()


@director.command(name='create-workspace')
@option('-p', '--director-path', required=True,
        help='The director path', type=ClickPath())
def create(director_path):
    """Create a director workspace."""
    if is_directory_traversal(director_path):
        click.echo('The director path is out of the openfl workspace scope.')
        sys.exit(1)
    director_path = Path(director_path).absolute()
    if director_path.exists():
        if not click.confirm('Director workspace already exists. Recreate?', default=True):
            sys.exit(1)
        shutil.rmtree(director_path)
    (director_path / 'cert').mkdir(parents=True, exist_ok=True)
    (director_path / 'logs').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORKSPACE / 'default/director.yaml', director_path / 'director.yaml')
