# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Envoy CLI."""

import logging
import shutil
from importlib import import_module
from pathlib import Path

import click
import sys
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from yaml import safe_load

from openfl.component.envoy.envoy import Envoy
from openfl.interface.cli_helper import WORKSPACE

logger = logging.getLogger(__name__)


@group()
@pass_context
def envoy(context):
    """Manage Federated Learning Envoy."""
    context.obj['group'] = 'envoy'


# @envoy.command(name='start')
# @option('-n', '--shard-name', required=True,
#         help='Current shard name')
# @option('-dh', '--director-host', required=True,
#         help='The FQDN of the federation director')
# @option('-dp', '--director-port', required=True,
#         help='The federation director port')
# @option('--tls/--disable-tls', default=True,
#         is_flag=True, help='Use TLS or not (By default TLS is enabled)')
# @option('-sc', '--shard-config-path', default='shard_config.yaml',
#         help='The shard config path', type=ClickPath(exists=True))
# @option('-rc', '--root-cert-path', 'root_certificate', default=None,
#         help='Path to a root CA cert')
# @option('-pk', '--private-key-path', 'private_key', default=None,
#         help='Path to a private key')
# @option('-oc', '--public-cert-path', 'certificate', default=None,
#         help='Path to a signed certificate')
def start_(shard_name, director_host, director_port, tls, shard_config_path,
           root_certificate, private_key, certificate):
    """Start the Envoy."""
    logger.info('🧿 Starting the Envoy.')

    shard_descriptor = shard_descriptor_from_config(shard_config_path)
    envoy = Envoy(
        shard_name=shard_name,
        director_host=director_host,
        director_port=director_port,
        shard_descriptor=shard_descriptor,
        tls=tls,
        root_certificate=root_certificate,
        private_key=private_key,
        certificate=certificate
    )

    envoy.start()


@envoy.command(name='create-workspace')
@option('-p', '--envoy-path', required=True,
        help='The Envoy path', type=ClickPath())
def create(envoy_path):
    """Create an envoy workspace."""
    envoy_path = Path(envoy_path)
    if envoy_path.exists():
        if not click.confirm('Envoy workspace already exists. Recreate?',
                             default=True):
            sys.exit(1)
        shutil.rmtree(envoy_path)
    (envoy_path / 'cert').mkdir(parents=True, exist_ok=True)
    (envoy_path / 'logs').mkdir(parents=True, exist_ok=True)
    (envoy_path / 'data').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORKSPACE / 'default/shard_config.yaml',
                    envoy_path / 'shard_config.yaml')
    shutil.copyfile(WORKSPACE / 'default/shard_descriptor.py',
                    envoy_path / 'shard_descriptor.py')
    shutil.copyfile(WORKSPACE / 'default/requirements.txt',
                    envoy_path / 'requirements.txt')


def shard_descriptor_from_config(shard_config_path: str):
    """Build a shard descriptor from config."""
    with open(shard_config_path) as stream:
        shard_config = safe_load(stream)
    template = shard_config.get('template')
    if not template:
        raise Exception(f'You should define a shard '
                        f'descriptor template in {shard_config_path}')
    class_name = template.split('.')[-1]
    module_path = '.'.join(template.split('.')[:-1])
    params = shard_config.get('params', {})

    module = import_module(module_path)
    instance = getattr(module, class_name)(**params)

    return instance


if __name__ == '__main__':
    start_(
        shard_name='env_one',
        director_host='localhost',
        director_port=50051,
        tls=False,
        shard_config_path='/home/dmitry/code/openfl/openfl-tutorials/interactive_api/Director_Pytorch_Kvasir_UNET/envoy_folder/shard_config.yaml',
        root_certificate=None,
        private_key=None,
        certificate=None,
    )
