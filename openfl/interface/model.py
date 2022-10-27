# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Plan module."""

from logging import getLogger

from click import echo
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath

logger = getLogger(__name__)


@group()
@pass_context
def model(context):
    """Manage Federated Learning Plans."""
    context.obj['group'] = 'model'


@model.command()
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
@option('-i', '--input', 'input_protobuf', required=True,
        help='The model protobuf to convert')
@option('-o', '--output', required=False,
        help='Filename the model will be saved to in native format',
        default='output_model')
def save(context, plan_config, cols_config, data_config, input_protobuf, output):
    """
    Save the model in native format (PyTorch / Tensorflow).
    """
    from pathlib import Path

    from openfl.federated import Plan
    from openfl.protocols import utils
    from openfl.utilities import split_tensor_dict_for_holdouts
    from openfl.utilities.utils import getfqdn_env
    from openfl.pipelines import NoCompressionPipeline

    plan = Plan.parse(plan_config_path=Path(plan_config),
                      cols_config_path=Path(cols_config),
                      data_config_path=Path(data_config))

    init_state_path = plan.config['aggregator']['settings']['init_state_path']

    collaborator_cname = list(plan.cols_data_paths)[0]

    data_loader = plan.get_data_loader(collaborator_cname)
    task_runner = plan.get_task_runner(data_loader)
    tensor_pipe = plan.get_tensor_pipe()

    logger.info(f'Loading OpenFL model protobuf:  🠆 {input_protobuf}')

    model = utils.load_proto(input_protobuf)

    tensor_dict, round_number = utils.deconstruct_model_proto(model, NoCompressionPipeline())

    # This may break for multiple models. task_runner.set_tensor_dict will need to handle multiple models
    task_runner.set_tensor_dict(tensor_dict)

    task_runner.save_native(output)

    logger.info(f'Saved model in native format:  🠆 {output}')
