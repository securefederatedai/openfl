# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Plan module."""

import sys
from logging import getLogger

from click import echo
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath

from openfl.utilities.path_check import is_directory_traversal

logger = getLogger(__name__)


@group()
@pass_context
def plan(context):
    """Manage Federated Learning Plans."""
    context.obj['group'] = 'plan'


@plan.command()
@pass_context
@option('-p', '--plan_config', required=False,
        help='Federated learning plan [plan/plan.yaml]',
        default='plan/plan.yaml', type=ClickPath(exists=True))
@option('-c', '--cols_config', required=False,
        help='Authorized collaborator list [plan/cols.yaml]',
        default='plan/cols.yaml', type=ClickPath(exists=True))
@option('-d', '--data_config', required=False,
        help='The data set/shard configuration file [plan/data.yaml]',
        default='plan/data.yaml')
@option('-a', '--aggregator_address', required=False,
        help='The FQDN of the federation agregator')
@option('-f', '--feature_shape', required=False,
        help='The input shape to the model')
def initialize(context, plan_config, cols_config, data_config,
               aggregator_address, feature_shape):
    """
    Initialize Data Science plan.

    Create a protocol buffer file of the initial model weights for
     the federation.
    """
    from pathlib import Path

    from openfl.experimental.federated import Plan
    from openfl.utilities.utils import getfqdn_env

    for p in [plan_config, cols_config, data_config]:
        if is_directory_traversal(p):
            echo(f'{p} is out of the openfl workspace scope.')
            sys.exit(1)

    plan_config = Path(plan_config).absolute()
    cols_config = Path(cols_config).absolute()
    data_config = Path(data_config).absolute()

    plan = Plan.parse(plan_config_path=plan_config,
                      cols_config_path=cols_config,
                      data_config_path=data_config)

    # TODO:  Is this part really needed?  Why would we need to collaborator
    #  name to know the input shape to the model?

    # if  feature_shape is None:
    #     if  cols_config is None:
    #         exit('You must specify either a feature
    #         shape or authorized collaborator
    #         list in order for the script to determine the input layer shape')

    plan_origin = Plan.parse(plan_config, resolve=False).config

    if (plan_origin['network']['settings']['agg_addr'] == 'auto'
            or aggregator_address):
        plan_origin['network']['settings']['agg_addr'] = aggregator_address or getfqdn_env()

        logger.warn(f'Patching Aggregator Addr in Plan'
                    f" 🠆 {plan_origin['network']['settings']['agg_addr']}")

        Plan.dump(plan_config, plan_origin)

    plan.config = plan_origin

    # Record that plan with this hash has been initialized
    if 'plans' not in context.obj:
        context.obj['plans'] = []
    context.obj['plans'].append(f'{plan_config.stem}_{plan.hash[:8]}')
    logger.info(f"{context.obj['plans']}")


def freeze_plan(plan_config):
    """Dump the plan to YAML file."""
    from pathlib import Path

    from openfl.experimental.federated import Plan

    plan = Plan()
    plan.config = Plan.parse(Path(plan_config), resolve=False).config

    Plan.dump(Path(plan_config), plan.config, freeze=True)
