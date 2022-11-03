# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Model CLI module."""

from click import group
from click import option
from click import pass_context
from click import Path as ClickPath
from logging import getLogger
from pathlib import Path

from openfl.federated import Plan
from openfl.federated import TaskRunner
from openfl.protocols import utils
from openfl.pipelines import NoCompressionPipeline

logger = getLogger(__name__)


@group()
@pass_context
def model(context):
    """Manage Federated Learning Models."""
    context.obj['group'] = 'model'


@model.command(name='save')
@pass_context
@option('-p', '--plan-config', required=False,
        help='Federated learning plan [plan/plan.yaml]',
        default='plan/plan.yaml', type=ClickPath(exists=True))
@option('-c', '--cols-config', required=False,
        help='Authorized collaborator list [plan/cols.yaml]',
        default='plan/cols.yaml', type=ClickPath(exists=True))
@option('-d', '--data-config', required=False,
        help='The data set/shard configuration file [plan/data.yaml]',
        default='plan/data.yaml', type=ClickPath(exists=True))
@option('-i', '--input', 'model_protobuf_path', required=True,
        help='The model protobuf to convert',
        type=ClickPath(exists=True))
@option('-o', '--output', 'output_filepath', required=False,
        help='Filename the model will be saved to in native format',
        default='output_model', type=ClickPath())
def save_(context, plan_config, cols_config, data_config, model_protobuf_path, output_filepath):
    """
    Save the model in native format (PyTorch / Keras).
    """

    task_runner = get_model(plan_config, cols_config, data_config, model_protobuf_path)

    output_filepath = Path(output_filepath).absolute()
    task_runner.save_native(output_filepath)
    logger.info(f'Saved model in native format:  🠆 {output_filepath}')


def get_model(
    plan_config: str,
    cols_config: str,
    data_config: str,
    model_protobuf_path: str
) -> TaskRunner:
    """
    Initialize TaskRunner and load it with provided model.pbuf.
    """

    model_protobuf_path = Path(model_protobuf_path).absolute()

    plan = Plan.parse(plan_config_path=Path(plan_config),
                      cols_config_path=Path(cols_config),
                      data_config_path=Path(data_config))

    collaborator_name = list(plan.cols_data_paths)[0]
    data_loader = plan.get_data_loader(collaborator_name)
    task_runner = plan.get_task_runner(data_loader=data_loader)

    logger.info(f'Loading OpenFL model protobuf:  🠆 {model_protobuf_path}')

    model_protobuf = utils.load_proto(model_protobuf_path)

    tensor_dict, _ = utils.deconstruct_model_proto(model_protobuf, NoCompressionPipeline())

    # This may break for multiple models.
    # task_runner.set_tensor_dict will need to handle multiple models
    task_runner.set_tensor_dict(tensor_dict, with_opt_vars=False)

    del task_runner.data_loader
    return task_runner
