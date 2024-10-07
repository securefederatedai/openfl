# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Plan module."""
import os
import sys
from logging import getLogger
from os import makedirs
from os.path import isfile
from pathlib import Path
from shutil import copyfile, rmtree
from subprocess import check_call  # nosec

from click import Path as ClickPath
from click import echo, group, option, pass_context
from yaml import FullLoader, dump, load

from openfl.federated import Plan
from openfl.interface.cli_helper import get_workspace_parameter
from openfl.protocols import utils
from openfl.utilities.click_types import InputSpec
from openfl.utilities.mocks import MockDataLoader
from openfl.utilities.path_check import is_directory_traversal
from openfl.utilities.split import split_tensor_dict_for_holdouts
from openfl.utilities.utils import getfqdn_env

logger = getLogger(__name__)


@group()
@pass_context
def plan(context):
    """Manage Federated Learning Plans.

    Args:
        context (click.core.Context): Click context.
    """
    context.obj["group"] = "plan"


@plan.command()
@pass_context
@option(
    "-p",
    "--plan_config",
    required=False,
    help="Federated learning plan [plan/plan.yaml]",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
)
@option(
    "-c",
    "--cols_config",
    required=False,
    help="Authorized collaborator list [plan/cols.yaml]",
    default="plan/cols.yaml",
    type=ClickPath(exists=True),
)
@option(
    "-d",
    "--data_config",
    required=False,
    help="The data set/shard configuration file [plan/data.yaml]",
    default="plan/data.yaml",
    type=ClickPath(exists=True),
)
@option(
    "-a",
    "--aggregator_address",
    required=False,
    help="The FQDN of the federation agregator",
)
@option(
    "-f",
    "--input_shape",
    cls=InputSpec,
    required=False,
    help="The input shape to the model. May be provided as a list:\n\n"
    "--input_shape [1,28,28]\n\n"
    "or as a dictionary for multihead models (must be passed in quotes):\n\n"
    "--input_shape \"{'input_0': [1, 240, 240, 4],'output_1': [1, 240, 240, 1]}\"\n\n ",
)
@option(
    "-g",
    "--gandlf_config",
    required=False,
    help="GaNDLF Configuration File Path",
)
@option(
    "-r",
    "--install_reqs",
    required=False,
    help="Install packages listed under 'requirements.txt'. True/False [Default: True]",
    default=True,
)
def initialize(
    context,
    plan_config,
    cols_config,
    data_config,
    aggregator_address,
    input_shape,
    gandlf_config,
    install_reqs,
):
    """Initialize Data Science plan.

    Create a protocol buffer file of the initial model weights for the
    federation.

    Args:
        context (click.core.Context): Click context.
        plan_config (str): Federated learning plan.
        cols_config (str): Authorized collaborator list.
        data_config (str): The data set/shard configuration file.
        aggregator_address (str): The FQDN of the federation aggregator.
        feature_shape (str): The input shape to the model.
        gandlf_config (str): GaNDLF Configuration File Path.
        install_reqs (bool): Whether to install packages listed under 'requirements.txt'.
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

    if install_reqs:
        requirements_filename = "requirements.txt"
        requirements_path = Path(requirements_filename).absolute()

        if isfile(f"{str(requirements_path)}"):
            check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    f"{str(requirements_path)}",
                ],
                shell=False,
            )
            echo(f"Successfully installed packages from {requirements_path}.")

            # Required to restart the process for newly installed packages to be recognized
            args_restart = [arg for arg in sys.argv if not arg.startswith("--install_reqs")]
            args_restart.append("--install_reqs=False")
            os.execv(args_restart[0], args_restart)
        else:
            echo("No additional requirements for workspace defined. Skipping...")

    plan = Plan.parse(
        plan_config_path=plan_config,
        cols_config_path=cols_config,
        data_config_path=data_config,
        gandlf_config_path=gandlf_config,
    )

    init_state_path = plan.config["aggregator"]["settings"]["init_state_path"]

    # This is needed to bypass data being locally available
    if input_shape is not None:
        logger.info(
            "Attempting to generate initial model weights with" f" custom input shape {input_shape}"
        )
        data_loader = MockDataLoader(input_shape)
    else:
        # If feature shape is not provided, data is assumed to be present
        collaborator_cname = list(plan.cols_data_paths)[0]
        data_loader = plan.get_data_loader(collaborator_cname)
    task_runner = plan.get_task_runner(data_loader)
    tensor_pipe = plan.get_tensor_pipe()

    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
        logger,
        task_runner.get_tensor_dict(False),
        **task_runner.tensor_dict_split_fn_kwargs,
    )

    logger.warn(
        f"Following parameters omitted from global initial model, "
        f"local initialization will determine"
        f" values: {list(holdout_params.keys())}"
    )

    model_snap = utils.construct_model_proto(
        tensor_dict=tensor_dict, round_number=0, tensor_pipe=tensor_pipe
    )

    logger.info("Creating Initial Weights File    🠆 %s", init_state_path)

    utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

    plan_origin = Plan.parse(
        plan_config_path=plan_config,
        gandlf_config_path=gandlf_config,
        resolve=False,
    )

    if plan_origin.config["network"]["settings"]["agg_addr"] == "auto" or aggregator_address:
        plan_origin.config["network"]["settings"]["agg_addr"] = aggregator_address or getfqdn_env()

        logger.warn(
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
        logger.info("Plan has not been initialized! Run 'fx plan" " initialize' before proceeding")
        return

    Plan.dump(Path(plan_config), plan.config, freeze=True)


@plan.command(name="freeze")
@option(
    "-p",
    "--plan_config",
    required=False,
    help="Federated learning plan [plan/plan.yaml]",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
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
    help="Name of the Federated learning plan",
    default="default",
    type=str,
)
def switch_(name):
    """Switch the current plan to this plan.

    Args:
        name (str): Name of the Federated learning plan.
    """
    switch_plan(name)


@plan.command(name="save")
@option(
    "-n",
    "--name",
    required=False,
    help="Name of the Federated learning plan",
    default="default",
    type=str,
)
def save_(name):
    """Save the current plan to this plan and switch.

    Args:
        name (str): Name of the Federated learning plan.
    """

    echo(f"Saving plan to {name}")
    # TODO: How do we get the prefix path? What happens if this gets executed
    #  outside of the workspace top directory?

    makedirs(f"plan/plans/{name}", exist_ok=True)
    copyfile("plan/plan.yaml", f"plan/plans/{name}/plan.yaml")

    switch_plan(name)  # Swtich the context


@plan.command(name="remove")
@option(
    "-n",
    "--name",
    required=False,
    help="Name of the Federated learning plan",
    default="default",
    type=str,
)
def remove_(name):
    """Remove this plan.

    Args:
        name (str): Name of the Federated learning plan.
    """

    if name != "default":
        echo(f"Removing plan {name}")
        # TODO: How do we get the prefix path? What happens if
        #  this gets executed outside of the workspace top directory?

        rmtree(f"plan/plans/{name}")

        switch_plan("default")  # Swtich the context back to the default

    else:
        echo("ERROR: Can't remove default plan")


@plan.command(name="print")
def print_():
    """Print the current plan."""

    current_plan_name = get_workspace_parameter("current_plan_name")
    echo(f"The current plan is: {current_plan_name}")
