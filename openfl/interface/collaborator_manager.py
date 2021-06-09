import logging

from click import group, option, pass_context
from click import Path as ClickPath
from openfl.component.collaborator_manager.collaborator_manager import CollaboratorManager

logger = logging.getLogger(__name__)


@group()
@pass_context
def collaborator_manager(context):
    """Manage Federated Learning Envoy."""
    context.obj['group'] = 'collaborator-manager'


@collaborator_manager.command(name='start')
@pass_context
@option('-n', '--shard-name', required=True,
        help='Current shard name')
@option('-d', '--director-uri', required=True,
        help='The FQDN of the federation director')
@option('-p', '--data-path', required=True,
        help='The data path', type=ClickPath(exists=True))
def start_(context, shard_name, director_uri, data_path):
    """Start the collaborator manager."""
    logger.info('🧿 Starting the Collaborator Manager.')
    keeper = CollaboratorManager(shard_name=shard_name, director_uri=director_uri)

    keeper.start(data_path)
