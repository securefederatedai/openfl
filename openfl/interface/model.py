# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Model CLI module."""
from logging import getLogger
from pathlib import Path

from click import Path as ClickPath
from click import confirm, group, option, pass_context, style

from openfl.federated import Plan
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import utils
from openfl.utilities.workspace import set_directory

logger = getLogger(__name__)


@group()
@pass_context
def model(context):
    """Manage Federated Learning Models.

    Args:
        context (click.core.Context): Click context.
    """
    context.obj["group"] = "model"


@model.command(name="save")
@pass_context
@option(
    "-i",
    "--input",
    "model_protobuf_path",
    required=True,
    help="The model protobuf to convert",
    type=ClickPath(exists=True),
)
@option(
    "-o",
    "--output",
    "output_filepath",
    required=False,
    help="Filename the model will be saved to in native format",
    default="output_model",
    type=ClickPath(writable=True),
)
@option(
    "-p",
    "--plan-config",
    required=False,
    help="Federated learning plan [plan/plan.yaml]",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
)
@option(
    "-c",
    "--cols-config",
    required=False,
    help="Authorized collaborator list [plan/cols.yaml]",
    default="plan/cols.yaml",
    type=ClickPath(exists=True),
)
@option(
    "-d",
    "--data-config",
    required=False,
    help="The data set/shard configuration file [plan/data.yaml]",
    default="plan/data.yaml",
    type=ClickPath(exists=True),
)
def save_(
    context,
    plan_config,
    cols_config,
    data_config,
    model_protobuf_path,
    output_filepath,
):
    """Save the model in native format (PyTorch / Keras).

    Args:
        context (click.core.Context): Click context.
        plan_config (str): Federated learning plan.
        cols_config (str): Authorized collaborator list.
        data_config (str): The data set/shard configuration file.
        model_protobuf_path (str): The model protobuf to convert.
        output_filepath (str): Filename the model will be saved to in native
            format.
    """
    output_filepath = Path(output_filepath).absolute()
    if output_filepath.exists():
        if not confirm(
            style(
                f"Do you want to overwrite the {output_filepath}?",
                fg="red",
                bold=True,
            )
        ):
            logger.info("Exiting")
            context.obj["fail"] = True
            return

    task_runner = get_model(plan_config, cols_config, data_config, model_protobuf_path)

    task_runner.save_native(output_filepath)
    logger.info("Saved model in native format:  🠆 %s", output_filepath)


def get_model(
    plan_config: str,
    cols_config: str,
    data_config: str,
    model_protobuf_path: str,
):
    """
    Initialize TaskRunner and load it with provided model.pbuf.

    Contrary to its name, this function returns a TaskRunner instance.
    The reason for this behavior is the flexibility of the TaskRunner
    interface and the diversity of the ways we store models in our template
    workspaces.

    Args:
        plan_config (str): Federated learning plan.
        cols_config (str): Authorized collaborator list.
        data_config (str): The data set/shard configuration file.
        model_protobuf_path (str): The model protobuf to convert.

    Returns:
        task_runner (instance): TaskRunner instance.
    """

    # Here we change cwd to the experiment workspace folder
    # because plan.yaml usually contains relative paths to components.
    workspace_path = Path(plan_config).resolve().parent.parent
    plan_config = Path(plan_config).resolve().relative_to(workspace_path)
    cols_config = Path(cols_config).resolve().relative_to(workspace_path)
    data_config = Path(data_config).resolve().relative_to(workspace_path)

    with set_directory(workspace_path):
        plan = Plan.parse(
            plan_config_path=plan_config,
            cols_config_path=cols_config,
            data_config_path=data_config,
        )
        collaborator_name = list(plan.cols_data_paths)[0]
        data_loader = plan.get_data_loader(collaborator_name)
        task_runner = plan.get_task_runner(data_loader=data_loader)

    model_protobuf_path = Path(model_protobuf_path).resolve()
    logger.info("Loading OpenFL model protobuf:  🠆 %s", model_protobuf_path)

    model_protobuf = utils.load_proto(model_protobuf_path)

    tensor_dict, _ = utils.deconstruct_model_proto(model_protobuf, NoCompressionPipeline())

    # This may break for multiple models.
    # task_runner.set_tensor_dict will need to handle multiple models
    task_runner.set_tensor_dict(tensor_dict, with_opt_vars=False)

    del task_runner.data_loader
    return task_runner
