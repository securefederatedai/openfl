# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl Native functions module.

This file defines openfl entrypoints to be used directly through python (not CLI)
"""
import os
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from sys import path

from flatten_json import flatten_preserve_lists

import openfl.interface.workspace as workspace
from openfl.federated import Plan
from openfl.interface.cli import setup_logging
from openfl.protocols import utils
from openfl.utilities import split_tensor_dict_for_holdouts

logger = getLogger(__name__)

WORKSPACE_PREFIX = os.path.join(os.path.expanduser('~'), '.local', 'workspace')


def create_collaborator(plan, name, model, client):
    """
    Create the collaborator.

    Using the same plan object to create multiple collaborators leads to
    identical collaborator objects. This function can be removed once
    collaborator generation is fixed in openfl/federated/plan/plan.py
    """
    return plan.get_collaborator(name, task_runner=model, client=client)


def get_plan(return_complete=False):
    """
    Return the flattened dictionary associated with the plan.

    To read the output in a human readable format, we recommend interpreting it
     as follows:

    ```
    print(json.dumps(fx.get_plan(), indent=4, sort_keys=True))
    ```

    Args:
        return_complete : bool (default=False)
            By default will not print the default file locations for each of
            the templates

    Returns:
        plan : dict
            flattened dictionary of the current plan
    """
    getLogger().setLevel('CRITICAL')

    plan_config = _setup_plan().config

    getLogger().setLevel('INFO')

    flattened_config = flatten_preserve_lists(plan_config, '.')[0]
    if not return_complete:
        keys_to_remove = [
            k for k, v in flattened_config.items()
            if ('defaults' in k or v is None)]
    else:
        keys_to_remove = [k for k, v in flattened_config.items() if v is None]
    for k in keys_to_remove:
        del flattened_config[k]

    return flattened_config


def init(workspace_template: str = 'default', log_level: str = 'debug'):
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
        workspace_template : str (default='default')
            The template that should be used as the basis for the experiment.
            Other options include are any of the template names [
            keras_cnn_mnist, tf_2dunet, tf_cnn_histology, mtorch_cnn_histology,
            torch_cnn_mnist]
        log_level: str (default='debug')
            Passed to setup_logging function to change default log level

    Returns:
        None
    """
    workspace.create(WORKSPACE_PREFIX, workspace_template)
    os.chdir(WORKSPACE_PREFIX)
    setup_logging(level=log_level)


def run_experiment(collaborator_dict: dict, override_config: dict = None, is_multi: bool = False,
                   max_workers: int = 0, mode: str = 'p=c*r'):
    """
    Core function that executes the FL Plan.

    Args:
        collaborator_dict : dict {collaborator_name(str): FederatedModel}
            This dictionary defines which collaborators will participate in the
            experiment, as well as a reference to that collaborator's
            federated model.
        override_config : dict {flplan.key : flplan.value}
            Override any of the plan parameters at runtime using this
            dictionary. To get a list of the available options, execute
            `fx.get_plan()`
        is_multi: bool
            If False collaborators will work in synchronous way
            If True each collaborator will be started in a separate process
        max_workers: int
            Used only when is_multi == True
            If max_workers > 0, max_worker == min(max_workers, len(collaborator_dict))
            If max_workers == 0, max_worker = len(collaborator_dict)
        mode: str
            Used only when is_multi == True
            If mode == 'p=c*r', collaborator will be created for every round and deleted after it
            If mode == 'p=c', each collaborator will be created one time and and their copy of
                the model will live in memory all the time.
            !!!WARNING!!! If mode == 'p=c' and max_workers < len(collaborator_dict), the first
            round will not be ended.

    Returns:
        final_federated_model : FederatedModel
            The final model resulting from the federated learning experiment
    """
    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(str(root))
    path.insert(0, str(work))

    # Update the plan if necessary
    if override_config and len(override_config) > 0:
        _update_plan(override_config)

    # TODO: Fix this implementation. The full plan parsing is reused here,
    # but the model and data will be overwritten based on user specifications
    plan = _setup_plan(save=False)

    # Overwrite plan values
    plan.authorized_cols = list(collaborator_dict)

    plan.runners_ = collaborator_dict
    rounds_to_train = plan.config['aggregator']['settings']['rounds_to_train']

    model = _get_model(collaborator_dict)
    init_state_path = plan.config['aggregator']['settings']['init_state_path']
    tensor_pipe = plan.get_tensor_pipe()

    _create_initial_weights_file(model, init_state_path, tensor_pipe)

    logger.info('Starting Experiment...')

    if not is_multi:
        last_tensor_dict = _run_sync_experiment(plan, collaborator_dict, rounds_to_train)
    else:
        last_tensor_dict = _run_multiprocess_experiment(plan, collaborator_dict, rounds_to_train,
                                                        max_workers, mode)

    model = _set_weights_for_the_final_model(model, rounds_to_train, last_tensor_dict)

    return model


def _create_initial_weights_file(model, init_state_path, tensor_pipe):
    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
        logger,
        model.get_tensor_dict(False),
    )
    model_snap = utils.construct_model_proto(
        tensor_dict=tensor_dict,
        round_number=0,
        tensor_pipe=tensor_pipe,
    )

    utils.dump_proto(dataobj=model_snap, fpath=init_state_path)
    logger.info(f'Creating Initial Weights File    🠆 {init_state_path}')


def _get_model(collaborator_dict):
    return deepcopy(next(iter(collaborator_dict.values())))


def _run_multiprocess_experiment(plan, collaborator_dict: dict, rounds_to_train: int,
                                 max_workers: int, mode: str = 'p=c'):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import set_start_method
    from multiprocessing.managers import BaseManager

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    if max_workers <= 0:
        max_workers = len(plan.authorized_cols)
    else:
        max_workers = min(max_workers, len(plan.authorized_cols))
    BaseManager.register('Aggregator', callable=plan.get_aggregator)
    with BaseManager() as manager:
        aggr = manager.Aggregator()
        with ProcessPoolExecutor(max_workers) as executor:
            if mode == 'p=c':
                _run_proc_eq_col(executor, plan, collaborator_dict, aggr, rounds_to_train)
            else:
                _run_proc_eq_col_x_round(executor, plan, collaborator_dict, aggr, rounds_to_train)
        return aggr.get_last_tensor_dict()


def _run_proc_eq_col(executor, plan, collaborator_dict, aggr, rounds_to_train):
    col_num = len(plan.authorized_cols)
    list(executor.map(_run_simulation_proc_eq_col,
                      [plan] * col_num,
                      (c for c in plan.authorized_cols),
                      (collaborator_dict[c] for c in plan.authorized_cols),
                      [aggr] * col_num,
                      [rounds_to_train] * col_num))


def _run_proc_eq_col_x_round(executor, plan, collaborator_dict, aggr, rounds_to_train):
    col_num = len(plan.authorized_cols)
    for _ in range(rounds_to_train):
        list(executor.map(_run_simulation_proc_eq_col_x_round,
                          [plan] * col_num,
                          (c for c in plan.authorized_cols),
                          (collaborator_dict[c] for c in plan.authorized_cols),
                          [aggr] * col_num))


def _run_simulation_proc_eq_col(p, name, model, a, r):
    col = create_collaborator(
        p, name, model, a
    )
    for _ in range(r):
        col.run_simulation()


def _run_simulation_proc_eq_col_x_round(p, name, model, a):
    col = create_collaborator(
        p, name, model, a
    )
    col.run_simulation()


def _run_sync_experiment(plan, collaborator_dict, rounds_to_train):
    aggr = plan.get_aggregator()
    collaborators = {
        c: create_collaborator(
            plan, c, collaborator_dict[c], aggr
        ) for c in plan.authorized_cols
    }

    for round_num in range(rounds_to_train):
        for col in plan.authorized_cols:
            collaborator = collaborators[col]
            collaborator.run_simulation()

    return aggr.get_last_tensor_dict()


def _setup_plan(save=True):
    """
    Dump the plan with all defaults + overrides set.

    Args:
        save : bool (default=True)
            Whether to save the plan to disk

    Returns:
        plan : Plan object
    """
    plan_config = 'plan/plan.yaml'
    cols_config = 'plan/cols.yaml'
    data_config = 'plan/data.yaml'
    plan = Plan.Parse(plan_config_path=Path(plan_config),
                      cols_config_path=Path(cols_config),
                      data_config_path=Path(data_config))
    if save:
        Plan.Dump(Path(plan_config), plan.config)

    return plan


def _set_weights_for_the_final_model(model, rounds_to_train, last_tensor_dict):
    model.set_optimizer_treatment('CONTINUE_LOCAL')
    model.rebuild_model(rounds_to_train - 1, last_tensor_dict, validation=True)
    return model


def _unflatten(config, separator='.'):
    """Unfold `config` settings that have `separator` in their names."""
    keys_to_separate = [k for k in config if separator in k]
    if len(keys_to_separate) > 0:
        for key in keys_to_separate:
            prefix = separator.join(key.split(separator)[:-1])
            suffix = key.split(separator)[-1]
            if prefix in config:
                temp = {**config[prefix], suffix: config[key]}
                config[prefix] = temp
            else:
                config[prefix] = {suffix: config[key]}
            del config[key]
        _unflatten(config, separator)
    return config


def _update_plan(override_config):
    """
    Update the plan with the provided override and save it to disk.

    For a list of available override options, call `fx.get_plan()`

    Args:
        override_config : dict {"COMPONENT.settings.variable" : value}

    Returns:
        None
    """
    plan_path = 'plan/plan.yaml'
    flat_plan_config = get_plan(return_complete=True)
    for k, v in override_config.items():
        if k in flat_plan_config:
            flat_plan_config[k] = v
            logger.info(f'Updating {k} to {v}... ')
        else:
            logger.info(f'Key {k} not found in plan. Ignoring... ')
    plan_config = _unflatten(flat_plan_config, '.')
    Plan.Dump(Path(plan_path), plan_config)
