# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

from openfl.callbacks.callback import Callback
from openfl.federated import Plan
from openfl.protocols import utils
from openfl.utilities.dataloading import initialize_minimal_dataloader
from openfl.utilities.workspace import set_directory

logger = logging.getLogger(__name__)


class ModelCheckpoint(Callback):
    """Save the model in native format at the end of the experiment.

    This callback saves the model in native format at the end of the experiment.
    It uses the `TaskRunner` to load the model from a protobuf file and then saves
    it in native format.

    Args:
        best_state_path (str): Path to the best model state file.
        last_state_path (str): Path to the last model state file.

    """

    def __init__(self, best_state_path, last_state_path):
        super().__init__()
        self.best_state_path = best_state_path
        self.last_state_path = last_state_path

    def on_experiment_end(self, logs=None):
        task_runner, tensor_pipe = initialize_task_runner()

        for state_path in [self.best_state_path, self.last_state_path]:
            state_path = Path(state_path).resolve()

            model_protobuf = utils.load_proto(state_path)

            tensor_dict, _ = utils.deconstruct_model_proto(model_protobuf, tensor_pipe)

            task_runner.set_tensor_dict(tensor_dict, with_opt_vars=False)
            state_path_no_ext = state_path.with_suffix("")
            output_filepath = task_runner.save_native(state_path_no_ext)
            logger.info("Model saved in native format at: %s", output_filepath)


def initialize_task_runner(
    plan_config: str = "plan/plan.yaml",
    cols_config: str = "plan/cols.yaml",
    data_config: str = "plan/data.yaml",
):
    """
    Initialize TaskRunner and load it with provided model.pbuf.

    Args:
        plan_config (str): Federated learning plan.
        cols_config (str): Authorized collaborator list.
        data_config (str): The data set/shard configuration file.

    Returns:
        task_runner (instance): TaskRunner instance.
        tensor_pipe (instance): TensorPipe instance.
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
            resolve=False,
        )
        data_loader = initialize_minimal_dataloader(plan)
        task_runner = plan.get_task_runner(data_loader=data_loader)
        tensor_pipe = plan.get_tensor_pipe()

    del task_runner.data_loader
    return task_runner, tensor_pipe
