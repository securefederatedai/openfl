# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Envoy CLI."""

import logging
import shutil
import sys
from importlib import import_module
from pathlib import Path

import click
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from yaml import safe_load

from openfl.component.envoy.envoy import Envoy
from openfl.interface.cli_helper import WORKSPACE
from openfl.utilities import click_types
from openfl.utilities.path_check import is_directory_traversal

logger = logging.getLogger(__name__)


@group()
@pass_context
def envoy(context):
    """Manage Federated Learning Envoy."""
    context.obj['group'] = 'envoy'


@envoy.command(name='start')
@option('-n', '--shard-name', required=True,
        help='Current shard name')
@option('-dh', '--director-host', required=True,
        help='The FQDN of the federation director', type=click_types.FQDN)
@option('-dp', '--director-port', required=True,
        help='The federation director port', type=click.IntRange(1, 65535))
@option('--tls/--disable-tls', default=True,
        is_flag=True, help='Use TLS or not (By default TLS is enabled)')
@option('-ec', '--envoy-config-path', default='envoy_config.yaml',
        help='The envoy config path', type=ClickPath(exists=True))
@option('-rc', '--root-cert-path', 'root_certificate', default=None,
        help='Path to a root CA cert', type=ClickPath(exists=True))
@option('-pk', '--private-key-path', 'private_key', default=None,
        help='Path to a private key', type=ClickPath(exists=True))
@option('-oc', '--public-cert-path', 'certificate', default=None,
        help='Path to a signed certificate', type=ClickPath(exists=True))
def start_(shard_name, director_host, director_port, tls, envoy_config_path,
           root_certificate, private_key, certificate):
    """Start the Envoy."""
    logger.info('🧿 Starting the Envoy.')
    if is_directory_traversal(shard_config_path):
        click.echo('The shard config path is out of the openfl workspace scope.')
        sys.exit(1)
        
    # Reed the Envoy config
    with open(envoy_config_path) as stream:
        envoy_config = safe_load(stream)

        # pass envoy parameters
    shard_config_path = Path(shard_config_path).absolute()
    if root_certificate:
        root_certificate = Path(root_certificate).absolute()
    if private_key:
        private_key = Path(private_key).absolute()
    if certificate:
        certificate = Path(certificate).absolute()

    envoy_params = envoy_config.get('params', {})
    for plugin_name, plugin_settings in envoy_params.get('optional_plugin_components', {}).items():
        template = plugin_settings.get('template')
        if not template:
            raise Exception('You should put a template'
                            f'for plugin {plugin_name}')
        module_path, _, class_name = template.rpartition('.')
        plugin_params = plugin_settings.get('params', {})

        module = import_module(module_path)
        instance = getattr(module, class_name)(**plugin_params)
        envoy_params[plugin_name] = instance

    shard_descriptor = shard_descriptor_from_config(envoy_config.get('shard_descriptor', {}))
    envoy = Envoy(
        shard_name=shard_name,
        director_host=director_host,
        director_port=director_port,
        shard_descriptor=shard_descriptor,
        tls=tls,
        root_certificate=root_certificate,
        private_key=private_key,
        certificate=certificate,
        **envoy_params
    )

    envoy.start()


@envoy.command(name='create-workspace')
@option('-p', '--envoy-path', required=True,
        help='The Envoy path', type=ClickPath())
def create(envoy_path):
    """Create an envoy workspace."""
    if is_directory_traversal(envoy_path):
        click.echo('The Envoy path is out of the openfl workspace scope.')
        sys.exit(1)
    envoy_path = Path(envoy_path).absolute()
    if envoy_path.exists():
        if not click.confirm('Envoy workspace already exists. Recreate?',
                             default=True):
            sys.exit(1)
        shutil.rmtree(envoy_path)
    (envoy_path / 'cert').mkdir(parents=True, exist_ok=True)
    (envoy_path / 'logs').mkdir(parents=True, exist_ok=True)
    (envoy_path / 'data').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORKSPACE / 'default/envoy_config.yaml',
                    envoy_path / 'envoy_config.yaml')
    shutil.copyfile(WORKSPACE / 'default/shard_descriptor.py',
                    envoy_path / 'shard_descriptor.py')
    shutil.copyfile(WORKSPACE / 'default/requirements.txt',
                    envoy_path / 'requirements.txt')


def shard_descriptor_from_config(shard_config: dict):
    """Build a shard descriptor from config."""
    template = shard_config.get('template')
    if not template:
        raise Exception('You should define a shard '
                        'descriptor template in the envoy config')
    class_name = template.split('.')[-1]
    module_path = '.'.join(template.split('.')[:-1])
    params = shard_config.get('params', {})

    module = import_module(module_path)
    instance = getattr(module, class_name)(**params)

    return instance
