import asyncio
import logging
import shutil
import sys
from pathlib import Path

import click
from click import Path as ClickPath
from click import group, pass_context
from click import option
from yaml import safe_load

from openfl.component.director.director import serve
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
