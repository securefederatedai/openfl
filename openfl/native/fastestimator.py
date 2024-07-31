# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""FederatedFastEstimator module."""
import os
from logging import getLogger
from pathlib import Path
from sys import path

import fastestimator as fe
from fastestimator.trace.io.best_model_saver import BestModelSaver

import openfl.native as fx
from openfl.federated import Plan
from openfl.federated.data import FastEstimatorDataLoader
from openfl.federated.task import FastEstimatorTaskRunner
from openfl.protocols import utils
from openfl.utilities.split import split_tensor_dict_for_holdouts


class FederatedFastEstimator:
    """A wrapper for fastestimator.estimator that allows running in federated
    mode.

    Attributes:
        estimator: The FastEstimator to be used.
        logger: A logger to record events.
        rounds: The number of rounds to train.
    """

    def __init__(self, estimator, override_config: dict = None, **kwargs):
        """Initializes a new instance of the FederatedFastEstimator class.

        Args:
            estimator: The FastEstimator to be used.
            override_config (dict, optional): A dictionary to override the
                default configuration. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.estimator = estimator
        self.logger = getLogger(__name__)
        fx.init(**kwargs)
        if override_config:
            fx.update_plan(override_config)

    def fit(self):
        """Runs the estimator in federated mode."""

        file = Path(__file__).resolve()
        # interface root, containing command modules
        root = file.parent.resolve()
        work = Path.cwd().resolve()

        path.append(str(root))
        path.insert(0, str(work))

        # TODO: Fix this implementation. The full plan parsing is reused here,
        # but the model and data will be overwritten based on
        # user specifications
        plan_config = Path(fx.WORKSPACE_PREFIX) / "plan" / "plan.yaml"
        cols_config = Path(fx.WORKSPACE_PREFIX) / "plan" / "cols.yaml"
        data_config = Path(fx.WORKSPACE_PREFIX) / "plan" / "data.yaml"

        plan = Plan.parse(
            plan_config_path=plan_config,
            cols_config_path=cols_config,
            data_config_path=data_config,
        )

        self.rounds = plan.config["aggregator"]["settings"]["rounds_to_train"]
        data_loader = FastEstimatorDataLoader(self.estimator.pipeline)
        runner = FastEstimatorTaskRunner(self.estimator, data_loader=data_loader)
        # Overwrite plan values
        tensor_pipe = plan.get_tensor_pipe()
        # Initialize model weights
        init_state_path = plan.config["aggregator"]["settings"]["init_state_path"]
        tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
            self.logger, runner.get_tensor_dict(False)
        )

        model_snap = utils.construct_model_proto(
            tensor_dict=tensor_dict, round_number=0, tensor_pipe=tensor_pipe
        )

        self.logger.info(f"Creating Initial Weights File" f"    🠆 {init_state_path}")

        utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

        self.logger.info("Starting Experiment...")

        aggregator = plan.get_aggregator()

        model_states = dict.fromkeys(plan.authorized_cols, None)
        runners = {}
        save_dir = {}
        data_path = 1
        for col in plan.authorized_cols:
            data = self.estimator.pipeline.data
            train_data, eval_data, test_data = split_data(
                data["train"],
                data["eval"],
                data["test"],
                data_path,
                len(plan.authorized_cols),
            )
            pipeline_kwargs = {}
            for k, v in self.estimator.pipeline.__dict__.items():
                if k in [
                    "batch_size",
                    "ops",
                    "num_process",
                    "drop_last",
                    "pad_value",
                    "collate_fn",
                ]:
                    pipeline_kwargs[k] = v
            pipeline_kwargs.update(
                {
                    "train_data": train_data,
                    "eval_data": eval_data,
                    "test_data": test_data,
                }
            )
            pipeline = fe.Pipeline(**pipeline_kwargs)

            data_loader = FastEstimatorDataLoader(pipeline)
            self.estimator.system.pipeline = pipeline

            runners[col] = FastEstimatorTaskRunner(
                estimator=self.estimator, data_loader=data_loader
            )
            runners[col].set_optimizer_treatment("CONTINUE_LOCAL")

            for trace in runners[col].estimator.system.traces:
                if isinstance(trace, BestModelSaver):
                    save_dir_path = f"{trace.save_dir}/{col}"
                    os.makedirs(save_dir_path, exist_ok=True)
                    save_dir[col] = save_dir_path

            data_path += 1

        # Create the collaborators
        collaborators = {
            collaborator: fx.create_collaborator(
                plan, collaborator, runners[collaborator], aggregator
            )
            for collaborator in plan.authorized_cols
        }

        model = None
        for round_num in range(self.rounds):
            for col in plan.authorized_cols:

                collaborator = collaborators[col]

                if round_num != 0:
                    # For FastEstimator Jupyter notebook, models must be
                    # saved in different directories (i.e. path must be
                    # reset here)

                    runners[col].estimator.system.load_state(f"save/{col}_state")
                    runners[col].rebuild_model(round_num, model_states[col])

                # Reset the save directory if BestModelSaver is present
                # in traces
                for trace in runners[col].estimator.system.traces:
                    if isinstance(trace, BestModelSaver):
                        trace.save_dir = save_dir[col]

                collaborator.run_simulation()

                model_states[col] = runners[col].get_tensor_dict(with_opt_vars=True)
                model = runners[col].model
                runners[col].estimator.system.save_state(f"save/{col}_state")

        # TODO This will return the model from the last collaborator,
        #  NOT the final aggregated model (though they should be similar).
        # There should be a method added to the aggregator that will load
        # the best model from disk and return it
        return model


def split_data(train, eva, test, rank, collaborator_count):
    """Split data into N parts, where N is the collaborator count.

    Args:
        train : The training data.
        eva : The evaluation data.
        test : The testing data.
        rank (int): The rank of the current collaborator.
        collaborator_count (int): The total number of collaborators.

    Returns:
        tuple: The training, evaluation, and testing data for the current
            collaborator.
    """
    if collaborator_count == 1:
        return train, eva, test

    fraction = [1.0 / float(collaborator_count)]
    fraction *= collaborator_count - 1

    # Expand the split list into individual parameters
    train_split = train.split(*fraction)
    eva_split = eva.split(*fraction)
    test_split = test.split(*fraction)

    train = [train]
    eva = [eva]
    test = [test]

    if type(train_split) is not list:
        train.append(train_split)
        eva.append(eva_split)
        test.append(test_split)
    else:
        # Combine all partitions into a single list
        train = [train] + train_split
        eva = [eva] + eva_split
        test = [test] + test_split

    # Extract the right shard
    train = train[rank - 1]
    eva = eva[rank - 1]
    test = test[rank - 1]

    return train, eva, test
