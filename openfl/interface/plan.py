# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Plan module."""

from socket import getfqdn
from logging import getLogger
from pathlib import Path
from click import Path as ClickPath
from click import group, option, pass_context
from click import echo

from openfl.protocols import utils
from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.federated import Plan
from openfl.interface.cli_helper import get_workspace_parameter

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
        default='plan/data.yaml', type=ClickPath(exists=True))
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
    plan = Plan.Parse(plan_config_path=Path(plan_config),
                      cols_config_path=Path(cols_config),
                      data_config_path=Path(data_config))

    init_state_path = plan.config['aggregator']['settings']['init_state_path']

    # TODO:  Is this part really needed?  Why would we need to collaborator
    #  name to know the input shape to the model?

    # if  feature_shape is None:
    #     if  cols_config is None:
    #         exit('You must specify either a feature
    #         shape or authorized collaborator
    #         list in order for the script to determine the input layer shape')
    print(plan.cols_data_paths)

    collaborator_cname = list(plan.cols_data_paths)[0]

    # else:

    #     logger.info(f'Using data object of type {type(data)}
    #     and feature shape {feature_shape}')
    #     raise NotImplementedError()

    # data_loader = plan.get_data_loader(collaborator_cname)
    # task_runner = plan.get_task_runner(collaborator_cname)

    # data_loader = plan.get_data_loader(collaborator_cname)
    task_runner = plan.get_task_runner(collaborator_cname)
    tensor_pipe = plan.get_tensor_pipe()

    # I believe there is no need for this line as task_runner has this variable
    # initialized with empty dict tensor_dict_split_fn_kwargs =
    # task_runner.tensor_dict_split_fn_kwargs or {}
    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
        logger,
        task_runner.get_tensor_dict(False),
        **task_runner.tensor_dict_split_fn_kwargs
    )

    logger.warn(f'Following parameters omitted from global initial model, '
                f'local initialization will determine'
                f' values: {list(holdout_params.keys())}')

    model_snap = utils.construct_model_proto(tensor_dict=tensor_dict,
                                             round_number=0,
                                             tensor_pipe=tensor_pipe)

    logger.info(f'Creating Initial Weights File    🠆 {init_state_path}')

    utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

    plan_origin = Plan.Parse(Path(plan_config), resolve=False).config

    if (plan_origin['network']['settings']['agg_addr'] == 'auto'
            or aggregator_address):
        plan_origin['network']['settings'] = plan_origin['network'].get(
            'settings', {})
        plan_origin['network']['settings']['agg_addr'] =\
            aggregator_address or getfqdn()

        logger.warn(f"Patching Aggregator Addr in Plan"
                    f" 🠆 {plan_origin['network']['settings']['agg_addr']}")

        Plan.Dump(Path(plan_config), plan_origin)

    plan.config = plan_origin

    # Record that plan with this hash has been initialized
    if 'plans' not in context.obj:
        context.obj['plans'] = []
    context.obj['plans'].append(f"{Path(plan_config).stem}_{plan.hash[:8]}")
    logger.info(f"{context.obj['plans']}")


def FreezePlan(plan_config):
    """Dump the plan to YAML file."""
    plan = Plan()
    plan.config = Plan.Parse(Path(plan_config), resolve=False).config

    init_state_path = plan.config['aggregator']['settings']['init_state_path']

    if not Path(init_state_path).exists():
        logger.info("Plan has not been initialized! Run 'fx plan"
                    " initialize' before proceeding")
        return

    Plan.Dump(Path(plan_config), plan.config, freeze=True)


@plan.command(name='freeze')
@pass_context
@option('-p', '--plan_config', required=False,
        help='Federated learning plan [plan/plan.yaml]',
        default='plan/plan.yaml', type=ClickPath(exists=True))
def freeze(context, plan_config):
    """
    Finalize the Data Science plan.

    Create a new plan file that embeds its hash in the file name
    (plan.yaml -> plan_{hash}.yaml) and changes the permissions to read only
    """
    FreezePlan(plan_config)


def switch_plan(name):
    """Switch the FL plan to this one."""
    from shutil import copyfile
    from os.path import isfile

    from yaml import load, dump, FullLoader

    plan_file = f'plan/plans/{name}/plan.yaml'
    if isfile(plan_file):

        echo(f'Switch plan to {name}')

        # Copy the new plan.yaml file to the top directory
        copyfile(plan_file, 'plan/plan.yaml')

        # Update the .workspace file to show the current workspace plan
        workspace_file = '.workspace'

        with open(workspace_file, 'r') as f:
            doc = load(f, Loader=FullLoader)

        if not doc:  # YAML is not correctly formatted
            doc = {}  # Create empty dictionary

        doc['current_plan_name'] = f'{name}'  # Switch with new plan name

        # Rewrite updated workspace file
        with open(workspace_file, 'w') as f:
            dump(doc, f)

    else:
        echo(f'Error: Plan {name} not found in plan/plans/{name}')


@plan.command(name='switch')
@pass_context
@option('-n', '--name', required=False,
        help='Name of the Federated learning plan',
        default='default', type=str)
def switch_(context, name):
    """Switch the current plan to this plan."""
    switch_plan(name)


@plan.command(name='save')
@pass_context
@option('-n', '--name', required=False,
        help='Name of the Federated learning plan',
        default='default', type=str)
def save_(context, name):
    """Save the current plan to this plan and switch."""
    from os import makedirs
    from shutil import copyfile

    echo(f'Saving plan to {name}')
    # TODO: How do we get the prefix path? What happens if this gets executed
    #  outside of the workspace top directory?

    makedirs(f'plan/plans/{name}', exist_ok=True)
    copyfile('plan/plan.yaml', f'plan/plans/{name}/plan.yaml')

    switch_plan(name)  # Swtich the context


@plan.command(name='remove')
@pass_context
@option('-n', '--name', required=False,
        help='Name of the Federated learning plan',
        default='default', type=str)
def remove_(context, name):
    """Remove this plan."""
    from shutil import rmtree

    if name != 'default':
        echo(f'Removing plan {name}')
        # TODO: How do we get the prefix path? What happens if
        #  this gets executed outside of the workspace top directory?

        rmtree(f'plan/plans/{name}')

        switch_plan('default')  # Swtich the context back to the default

    else:
        echo("ERROR: Can't remove default plan")


@plan.command(name='print')
def print_():
    """Print the current plan."""
    current_plan_name = get_workspace_parameter("current_plan_name")
    echo(f'The current plan is: {current_plan_name}')
