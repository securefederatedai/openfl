import asyncio
import logging

from click import group, pass_context
from click import option
from openfl.component.director.director import serve

logger = logging.getLogger(__name__)


@group()
@pass_context
def director(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'director'


@director.command(name='start')
@pass_context
@option('--sample-shape', '-ss', multiple=True,
        help='Sample shape')
@option('--target-shape', '-ts', multiple=True,
        help='Target shape')
def start_(context, sample_shape, target_shape):
    """Start the director service."""
    logger.info('🧿 Starting the Director Service.')
    logger.info(F'Sample shape: {sample_shape}, target shape: {target_shape}')
    asyncio.run(serve(sample_shape=list(sample_shape), target_shape=list(target_shape)))
