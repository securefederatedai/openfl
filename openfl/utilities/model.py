# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from pathlib import Path

from openfl.federated import Plan
from openfl.utilities.dataloading import get_dataloader
from openfl.utilities.workspace import set_directory

logger = getLogger(__name__)


def get_model(
    plan_config: str = "plan/plan.yaml",
    cols_config: str = "plan/cols.yaml",
    data_config: str = "plan/data.yaml",
):
    """
    Initialize TaskRunner.

    Contrary to its name, this function returns a TaskRunner instance, which 
    contains the model architecture definition used in the experiment.
    The reason for this behavior is the flexibility of the TaskRunner
    interface and the diversity of the ways we store models in our template
    workspaces.

    Args:
        plan_config (str): Federated learning plan. Defaults to 'plan/cols.yaml'.
        cols_config (str): Authorized collaborator list. Defaults to 'plan/plan.yaml'.
        data_config (str): The data set/shard configuration file. Defaults to 'plan/data.yaml'.

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
        data_loader = get_dataloader(plan, prefer_minimal=True)
        task_runner = plan.get_task_runner(data_loader=data_loader)

    del task_runner.data_loader
    return task_runner
