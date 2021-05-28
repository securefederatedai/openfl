import asyncio
import logging

from click import group, pass_context
from openfl.services.director import serve

logger = logging.getLogger(__name__)


@group()
@pass_context
def director(context):
    """Manage Federated Learning Director."""
    context.obj['group'] = 'director'


@director.command(name='start')
@pass_context
def start_(context):
    """Start the aggregator service."""
    logger.info('🧿 Starting the Director Service.')
    asyncio.run(serve(sample_shape=1, target_shape=1))
