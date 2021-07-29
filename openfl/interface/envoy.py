# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Collaborator manager CLI."""

import logging
import shutil
import sys
from importlib import import_module
from os import path
from pathlib import Path

import click
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from yaml import safe_load

from openfl.component.envoy.envoy import Envoy
from openfl.interface.cli_helper import SITEPACKS
from openfl.interface.cli_helper import WORKSPACE

logger = logging.getLogger(__name__)


@group()
@pass_context
def envoy(context):
    """Manage Federated Learning Envoy."""
    context.obj['group'] = 'collaborator-manager'


@envoy.command(name='start')
@option('-n', '--shard-name', required=True,
        help='Current shard name')
@option('-d', '--director-uri', required=True,
        help='The FQDN of the federation director')
@option('--disable-tls', default=False,
        is_flag=True)
@option('-sc', '--shard-config-path', default='shard_config.yaml',
        help='The shard config path', type=ClickPath(exists=True))
@option('-rc', '--root-cert-path', 'root_ca', default=None,
        help='Path to a root CA cert')
@option('-pk', '--private-key-path', 'key', default=None,
        help='Path to a private key')
@option('-oc', '--public-cert-path', 'cert', default=None,
        help='Path to a signed certificate')
def start_(shard_name, director_uri, disable_tls, shard_config_path,
           root_ca, key, cert):
    """Start the Envoy."""
    logger.info('🧿 Starting the Envoy.')

    shard_descriptor = shard_descriptor_from_config(shard_config_path)
    envoy = Envoy(
        shard_name=shard_name,
        director_uri=director_uri,
        shard_descriptor=shard_descriptor,
        disable_tls=disable_tls,
        root_ca=root_ca,
        key=key,
        cert=cert
    )

    envoy.start()


@envoy.command(name='create-workspace')
@option('-p', '--envoy-path', required=True,
        help='The Envoy path', type=ClickPath())
def create(envoy_path):
    """Create a envoy workspace."""
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
    class_name = path.splitext(shard_config['template'])[1].strip('.')
    module_path = path.splitext(shard_config['template'])[0]
    params = shard_config['params']

    module = import_module(module_path)
    instance = getattr(module, class_name)(**params)

    return instance
