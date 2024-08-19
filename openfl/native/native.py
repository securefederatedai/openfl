# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""OpenFL Native functions module.

This file defines openfl entrypoints to be used directly through python (not
CLI)
"""
import importlib
import json
import logging
import os
from copy import copy
from logging import basicConfig, getLogger
from pathlib import Path
from sys import path

import flatten_json
from rich.console import Console
from rich.logging import RichHandler

import openfl.interface.aggregator as aggregator
import openfl.interface.collaborator as collaborator
import openfl.interface.workspace as workspace
from openfl.federated import Plan
from openfl.protocols import utils
from openfl.utilities import add_log_level
from openfl.utilities.split import split_tensor_dict_for_holdouts

logger = getLogger(__name__)

WORKSPACE_PREFIX = os.path.join(os.path.expanduser("~"), ".local", "workspace")


def setup_plan(log_level="CRITICAL"):
    """
    Dump the plan with all defaults + overrides set.

    Args:
        log_level (str, optional): The log level Whether to save the plan to
            disk.
        Defaults to 'CRITICAL'.

    Returns:
        plan: Plan object.
    """
    plan_config = "plan/plan.yaml"
    cols_config = "plan/cols.yaml"
    data_config = "plan/data.yaml"

    current_level = logging.root.level
    getLogger().setLevel(log_level)
    plan = Plan.parse(
        plan_config_path=Path(plan_config),
        cols_config_path=Path(cols_config),
        data_config_path=Path(data_config),
        resolve=False,
    )
    getLogger().setLevel(current_level)

    return plan


def flatten(config, return_complete=False):
    """
    Flatten nested config.

    Args:
        config (dict): The configuration dictionary to flatten.
        return_complete (bool, optional): Whether to return the complete
            flattened config. Defaults to False.

    Returns:
        flattened_config (dict): The flattened configuration dictionary.
    """
    flattened_config = flatten_json.flatten(config, ".")
    if not return_complete:
        keys_to_remove = [k for k, v in flattened_config.items() if ("defaults" in k or v is None)]
    else:
        keys_to_remove = [k for k, v in flattened_config.items() if v is None]
    for k in keys_to_remove:
        del flattened_config[k]

    return flattened_config


def update_plan(override_config, plan=None, resolve=True):
    """Updates the plan with the provided override and saves it to disk.

    For a list of available override options, call `fx.get_plan()`

    Args:
        override_config (dict): A dictionary of values to override in the plan.
        plan (Plan, optional): The plan to update. If None, a new plan is set
            up. Defaults to None.
        resolve (bool, optional): Whether to resolve the plan. Defaults to
            True.

    Returns:
        plan (object): The updated plan.
    """
    if plan is None:
        plan = setup_plan()
    flat_plan_config = flatten(plan.config, return_complete=True)

    org_list_keys_with_count = {}
    for k in flat_plan_config:
        k_split = k.rsplit(".", 1)
        if k_split[1].isnumeric():
            if k_split[0] in org_list_keys_with_count:
                org_list_keys_with_count[k_split[0]] += 1
            else:
                org_list_keys_with_count[k_split[0]] = 1

    for key, val in override_config.items():
        if key in org_list_keys_with_count:
            # remove old list corresponding to this key entirely
            for idx in range(org_list_keys_with_count[key]):
                del flat_plan_config[f"{key}.{idx}"]
            logger.info("Updating %s to %s... ", key, val)
        elif key in flat_plan_config:
            logger.info("Updating %s to %s... ", key, val)
        else:
            # TODO: We probably need to validate the new key somehow
            logger.info(
                "Did not find %s in config. Make sure it should exist. Creating...",
                key,
            )
        if type(val) is list:
            for idx, v in enumerate(val):
                flat_plan_config[f"{key}.{idx}"] = v
        else:
            flat_plan_config[key] = val

    plan.config = unflatten(flat_plan_config, ".")
    if resolve:
        plan.resolve()
    return plan


def unflatten(config, separator="."):
    """Unfolds `config` settings that have `separator` in their names.

    Args:
        config (dict): The flattened configuration dictionary to unfold.
        separator (str, optional): The separator used in the flattened config.
            Defaults to '.'.

    Returns:
        config (dict): The unfolded configuration dictionary.
    """
    config = flatten_json.unflatten_list(config, separator)
    return config


def setup_logging(level="INFO", log_file=None):
    """Initializes logging settings.

    Args:
        level (str, optional): The log level. Defaults to 'INFO'.
        log_file (str, optional): The name of the file to log to.
        If None, logs are not saved to a file. Defaults to None.
    """
    # Setup logging

    if importlib.util.find_spec("tensorflow") is not None:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    metric = 25
    add_log_level("METRIC", metric)

    if isinstance(level, str):
        level = level.upper()

    handlers = []
    if log_file:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s %(filename)s:%(lineno)d"
        )
        fh.setFormatter(formatter)
        handlers.append(fh)

    console = Console(width=160)
    handlers.append(RichHandler(console=console))
    basicConfig(level=level, format="%(message)s", datefmt="[%X]", handlers=handlers)


def init(
    workspace_template: str = "default",
    log_level: str = "INFO",
    log_file: str = None,
    agg_fqdn: str = None,
    col_names=None,
):
    """
    Initialize the openfl package.

    It performs the following tasks:

        1. Creates a workspace in ~/.local/workspace (Equivalent to `fx
        workspace create --prefix ~/.local/workspace --template
        $workspace_template)
        2. Setup certificate authority (equivalent to `fx workspace certify`)
        3. Setup aggregator PKI (equivalent to `fx aggregator
        generate-cert-request` followed by `fx aggregator certify`)
        4. Setup list of collaborators (col_names) and their PKI. (Equivalent
        to running `fx collaborator generate-cert-request` followed by `fx
        collaborator certify` for each of the collaborators in col_names)
        5. Setup logging

    Args:
        workspace_template (str): The template that should be used as the
            basis for the experiment.  Defaults to 'default'.
            Other options include are any of the template names
            [keras_cnn_mnist, tf_2dunet, tf_cnn_histology,
            mtorch_cnn_histology, torch_cnn_mnist].
        log_level (str): Log level for logging. METRIC level is available.
            Defaults to 'INFO'.
        log_file (str): Name of the file in which the log will be duplicated.
            If None, logs are not saved to a file. Defaults to None.
        agg_fqdn (str): The local node's fully qualified domain name (if it
            can't be resolved automatically). Defaults to None.
        col_names (list[str]): The names of the collaborators that will be
            created. These collaborators will be set up to participate in the
            experiment, but are not required to. Defaults to None.

    Returns:
        None
    """
    if col_names is None:
        col_names = ["one", "two"]
    workspace.create(WORKSPACE_PREFIX, workspace_template)
    os.chdir(WORKSPACE_PREFIX)
    workspace.certify()
    aggregator.generate_cert_request(agg_fqdn)
    aggregator.certify(agg_fqdn, silent=True)
    data_path = 1
    for col_name in col_names:
        collaborator.create(col_name, str(data_path), silent=True)
        collaborator.generate_cert_request(col_name, silent=True, skip_package=True)
        collaborator.certify(col_name, silent=True)
        data_path += 1

    setup_logging(level=log_level, log_file=log_file)


def get_collaborator(plan, name, model, aggregator):
    """Create the collaborator.

    Using the same plan object to create multiple collaborators leads to
    identical collaborator objects. This function can be removed once
    collaborator generation is fixed in openfl/federated/plan/plan.py

    Args:
        plan (Plan): The plan to use to create the collaborator.
        name (str): The name of the collaborator.
        model (Model): The model to use for the collaborator.
        aggregator (Aggregator): The aggregator to use for the collaborator.

    Returns:
        Collaborator: The created collaborator.
    """
    plan = copy(plan)

    return plan.get_collaborator(name, task_runner=model, client=aggregator)


def run_experiment(collaborator_dict: dict, override_config: dict = None):
    """Core function that executes the FL Plan.

    Args:
        collaborator_dict (dict): A dictionary mapping collaborator names to
            their federated models.
            Example: {collaborator_name(str): FederatedModel}
            This dictionary defines which collaborators will participate in
            the experiment, as well as a reference to that collaborator's
            federated model.
        override_config (dict, optional): A dictionary of values to override
            in the plan. Defaults to None.
            Example: dict {flplan.key : flplan.value}
            Override any of the plan parameters at runtime using this
            dictionary. To get a list of the available options, execute
            `fx.get_plan()`

    Returns:
        model: Final Federated model. The model resulting from the federated
            learning experiment
    """

    if override_config is None:
        override_config = {}

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(str(root))
    path.insert(0, str(work))

    # Update the plan if necessary
    plan = update_plan(override_config)
    # Overwrite plan values
    plan.authorized_cols = list(collaborator_dict)
    tensor_pipe = plan.get_tensor_pipe()

    # This must be set to the final index of the list (this is the last
    # tensorflow session to get created)
    plan.runner_ = list(collaborator_dict.values())[-1]
    model = plan.runner_

    # Initialize model weights
    init_state_path = plan.config["aggregator"]["settings"]["init_state_path"]
    rounds_to_train = plan.config["aggregator"]["settings"]["rounds_to_train"]
    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
        logger, model.get_tensor_dict(False)
    )

    model_snap = utils.construct_model_proto(
        tensor_dict=tensor_dict, round_number=0, tensor_pipe=tensor_pipe
    )

    logger.info("Creating Initial Weights File    🠆 %s", init_state_path)

    utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

    logger.info("Starting Experiment...")

    aggregator = plan.get_aggregator()

    # get the collaborators
    collaborators = {
        collaborator: get_collaborator(
            plan, collaborator, collaborator_dict[collaborator], aggregator
        )
        for collaborator in plan.authorized_cols
    }

    for _ in range(rounds_to_train):
        for col in plan.authorized_cols:
            collaborator = collaborators[col]
            collaborator.run_simulation()

    # Set the weights for the final model
    model.rebuild_model(rounds_to_train - 1, aggregator.last_tensor_dict, validation=True)
    return model


def get_plan(fl_plan=None, indent=4, sort_keys=True):
    """Returns a string representation of the current Plan.

    Args:
        fl_plan (Plan): The plan to get a string representation of. If None, a
            new plan is set up. Defaults to None.
        indent (int): The number of spaces to use for indentation in the
            string representation. Defaults to 4.
        sort_keys (bool): Whether to sort the keys in the string
            representation. Defaults to True.

    Returns:
        str: A string representation of the plan.
    """
    if fl_plan is None:
        plan = setup_plan()
    else:
        plan = fl_plan
    flat_plan_config = flatten(plan.config)
    return json.dumps(flat_plan_config, indent=indent, sort_keys=sort_keys)
