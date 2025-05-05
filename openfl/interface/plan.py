# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Plan module."""

import sys
from logging import getLogger
from os import makedirs
from os.path import isfile, splitext
from pathlib import Path
from shutil import copyfile, rmtree

from click import Path as ClickPath
from click import echo, group, option, pass_context
from yaml import FullLoader, dump, load

from openfl.federated import Plan
from openfl.interface.cli_helper import get_workspace_parameter
from openfl.protocols import utils
from openfl.utilities.click_types import InputSpec
from openfl.utilities.dataloading import get_dataloader
from openfl.utilities.path_check import is_directory_traversal
from openfl.utilities.split import split_tensor_dict_for_holdouts
from openfl.utilities.utils import getfqdn_env

logger = getLogger(__name__)


@group()
@pass_context
def plan(context):
    """Manage Federated Learning Plans."""
    context.obj["group"] = "plan"


@plan.command()
@pass_context
@option(
    "-p",
    "--plan_config",
    required=False,
    help="Path to an FL plan.",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
@option(
    "-c",
    "--cols_config",
    required=False,
    help="Path to an authorized collaborator list.",
    default="plan/cols.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
@option(
    "-d",
    "--data_config",
    required=False,
    help="The dataset shard configuration file.",
    default="plan/data.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
@option(
    "-a",
    "--aggregator_address",
    required=False,
    help="The FQDN of the federation aggregator",
)
@option(
    "-f",
    "--input_shape",
    cls=InputSpec,
    required=False,
    help="""
    The input spec of the model.

    May be provided as a list for single input head: ``--input-shape [3,32,32]``,

    or as a dictionary for multihead models (must be passed in quotes):

    ``--input-shape "{'input_0': [1, 240, 240, 4],'input_1': [1, 240, 240, 1]}"``.
    """,
)
@option(
    "-g",
    "--gandlf_config",
    required=False,
    help="GaNDLF Configuration File Path",
)
@option(
    "-i",
    "--init_model_path",
    required=False,
    help="Path to initial model. It can be a protobuf or native format.",
    type=ClickPath(exists=True),
)
def initialize(
    context,
    plan_config,
    cols_config,
    data_config,
    aggregator_address,
    input_shape,
    gandlf_config,
    init_model_path,
):
    """
    Initializes a Data Science plan and generates a protobuf file of the
    initial model weights for the federation.
    """

    for p in [plan_config, cols_config, data_config]:
        if is_directory_traversal(p):
            echo(f"{p} is out of the openfl workspace scope.")
            sys.exit(1)

    plan_config = Path(plan_config).absolute()
    cols_config = Path(cols_config).absolute()
    data_config = Path(data_config).absolute()
    if gandlf_config is not None:
        gandlf_config = Path(gandlf_config).absolute()

    plan = Plan.parse(
        plan_config_path=plan_config,
        cols_config_path=cols_config,
        data_config_path=data_config,
        gandlf_config_path=gandlf_config,
    )

    if "connector" in plan.config:
        logger.info("OpenFL Connector enabled: %s", plan.config["connector"])
        # Only need to initialize task runner to install apps/packages
        # that were not installable via requirements.txt
        plan.get_task_runner(data_loader=None)
    else:
        init_state_path = plan.config["aggregator"]["settings"]["init_state_path"]
        # This is needed to bypass data being locally available
        if input_shape is not None:
            logger.info(
                "Attempting to generate initial model weights with custom input shape "
                f"{input_shape}"
            )

        # Initialize tensor dictionary
        init_tensor_dict, task_runner, round_number = _initialize_tensor_dict(
            plan, input_shape, init_model_path
        )

        tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
            init_tensor_dict,
            **task_runner.tensor_dict_split_fn_kwargs,
        )

        logger.warning(
            f"Following parameters omitted from global initial model, "
            f"local initialization will determine"
            f" values: {list(holdout_params.keys())}"
        )

        # Save the model state
        try:
            logger.info(f"Saving model state to {init_state_path}")
            plan.save_model_to_state_file(
                tensor_dict=tensor_dict, round_number=round_number, output_path=init_state_path
            )
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")
            raise

    plan_origin = Plan.parse(
        plan_config_path=plan_config,
        gandlf_config_path=gandlf_config,
        resolve=False,
    )

    if plan_origin.config["network"]["settings"]["agg_addr"] == "auto" or aggregator_address:
        plan_origin.config["network"]["settings"]["agg_addr"] = aggregator_address or getfqdn_env()

        logger.warning(
            f"Patching Aggregator Addr in Plan"
            f" 🠆 {plan_origin.config['network']['settings']['agg_addr']}"
        )

        Plan.dump(plan_config, plan_origin.config)

    if gandlf_config is not None:
        Plan.dump(plan_config, plan_origin.config)

    # Record that plan with this hash has been initialized
    if "plans" not in context.obj:
        context.obj["plans"] = []
    context.obj["plans"].append(f"{plan_config.stem}_{plan_origin.hash[:8]}")
    logger.info(f"{context.obj['plans']}")


def _initialize_tensor_dict(plan, input_shape, init_model_path):
    """Initialize and return the tensor dictionary.

    Args:
        plan: The federation plan object
        input_shape: The input shape to the model
        init_model_path: Path to initial model. It can be a protobuf or native format."

    Returns:
        Tuple of (tensor_dict, task_runner, round_number)
    """
    data_loader = get_dataloader(plan, prefer_minimal=True, input_shape=input_shape)
    task_runner = plan.get_task_runner(data_loader)
    tensor_pipe = plan.get_tensor_pipe()
    round_number = 0

    if init_model_path and isfile(init_model_path):
        logger.info(f"Loading initial model from {init_model_path}")
        file_extension = splitext(init_model_path)[1]

        if file_extension == ".pbuf":
            model_proto = utils.load_proto(init_model_path)
            init_tensor_dict, round_number = utils.deconstruct_model_proto(model_proto, tensor_pipe)
        else:
            try:
                task_runner.load_native(init_model_path)
                init_tensor_dict = task_runner.get_tensor_dict(False)
            except Exception as e:
                logger.error(f"Failed to load native model: {e}")
                raise
    else:
        init_tensor_dict = task_runner.get_tensor_dict(False)

    return init_tensor_dict, task_runner, round_number


# TODO: looks like Plan.method
def freeze_plan(plan_config):
    """Dump the plan to YAML file.

    Args:
        plan_config (str): Federated learning plan.
    """

    plan = Plan()
    plan.config = Plan.parse(Path(plan_config), resolve=False).config

    init_state_path = plan.config["aggregator"]["settings"]["init_state_path"]

    if not Path(init_state_path).exists():
        logger.info("Plan has not been initialized! Run 'fx plan initialize' before proceeding")
        return

    Plan.dump(Path(plan_config), plan.config, freeze=True)


@plan.command(name="freeze")
@option(
    "-p",
    "--plan_config",
    required=False,
    help="Path to an FL plan.",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
def freeze(plan_config):
    """Finalize the Data Science plan.

    Create a new plan file that embeds its hash in the file name
    (plan.yaml -> plan_{hash}.yaml) and changes the permissions to read only.

    Args:
        plan_config (str): Federated learning plan.
    """
    if is_directory_traversal(plan_config):
        echo("Plan config path is out of the openfl workspace scope.")
        sys.exit(1)
    freeze_plan(plan_config)


def switch_plan(name):
    """Switch the FL plan to this one.

    Args:
        name (str): Name of the Federated learning plan.
    """

    plan_file = f"plan/plans/{name}/plan.yaml"
    if isfile(plan_file):
        echo(f"Switch plan to {name}")

        # Copy the new plan.yaml file to the top directory
        copyfile(plan_file, "plan/plan.yaml")

        # Update the .workspace file to show the current workspace plan
        workspace_file = ".workspace"

        with open(workspace_file, "r", encoding="utf-8") as f:
            doc = load(f, Loader=FullLoader)

        if not doc:  # YAML is not correctly formatted
            doc = {}  # Create empty dictionary

        doc["current_plan_name"] = f"{name}"  # Switch with new plan name

        # Rewrite updated workspace file
        with open(workspace_file, "w", encoding="utf-8") as f:
            dump(doc, f)

    else:
        echo(f"Error: Plan {name} not found in plan/plans/{name}")


@plan.command(name="switch")
@option(
    "-n",
    "--name",
    required=False,
    help="Name of the FL plan to switch to.",
    default="default",
    type=str,
)
def switch_(name):
    """Switch the current plan to this plan."""
    switch_plan(name)


@plan.command(name="save")
@option(
    "-n",
    "--name",
    required=False,
    help="Name of the FL plan.",
    default="default",
    type=str,
)
def save_(name):
    """Saves the given plan and switches to it."""

    echo(f"Saving plan to {name}")
    # TODO: How do we get the prefix path? What happens if this gets executed
    #  outside of the workspace top directory?

    makedirs(f"plan/plans/{name}", exist_ok=True)
    copyfile("plan/plan.yaml", f"plan/plans/{name}/plan.yaml")

    switch_plan(name)  # Switch the context


@plan.command(name="remove")
@option(
    "-n",
    "--name",
    required=False,
    help="Name of the FL plan to remove.",
    default="default",
    type=str,
    show_default=True,
)
def remove_(name):
    """Removes given plan."""

    if name != "default":
        echo(f"Removing plan {name}")
        # TODO: How do we get the prefix path? What happens if
        #  this gets executed outside of the workspace top directory?

        rmtree(f"plan/plans/{name}")

        switch_plan("default")  # Switch the context back to the default

    else:
        echo("ERROR: Can't remove default plan")


@plan.command(name="show")
def show_():
    """Shows the active plan."""
    current_plan_name = get_workspace_parameter("current_plan_name")
    echo(f"The current plan is: {current_plan_name}")
