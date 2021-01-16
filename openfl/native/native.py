# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl Native functions module.

This file defines openfl entrypoints to be used directly through python (not CLI)
"""

import os
from logging import getLogger
from pathlib import Path
from copy import copy
from flatten_json import flatten_preserve_lists
import openfl.interface.workspace as workspace
import openfl.interface.aggregator as aggregator
import openfl.interface.collaborator as collaborator

from openfl.federated import Plan

from openfl.protocols import utils
from openfl.utilities import split_tensor_dict_for_holdouts

logger = getLogger(__name__)

WORKSPACE_PREFIX = os.path.join(os.path.expanduser('~'), '.local', 'workspace')


def setup_plan(save=True):
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

    plan_config = setup_plan().config

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


def update_plan(override_config):
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
    plan_config = unflatten(flat_plan_config, '.')
    Plan.Dump(Path(plan_path), plan_config)


def unflatten(config, separator='.'):
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
        unflatten(config, separator)
    return config


def setup_logging():
    """Initialize logging settings."""
    # Setup logging
    from logging import basicConfig
    from rich.console import Console
    from rich.logging import RichHandler
    import pkgutil
    if True if pkgutil.find_loader('tensorflow') else False:
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    console = Console(width=160)
    basicConfig(
        level='INFO',
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(console=console)]
    )


def init(workspace_template='default', agg_fqdn=None, col_names=['one', 'two']):
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
        agg_fqdn : str
           The local node's fully qualified domain name (if it can't be
           resolved automatically)
        col_names: list[str]
           The names of the collaborators that will be created. These
           collaborators will be set up to participate in the experiment, but
           are not required to

    Returns:
        None
    """
    workspace.create(WORKSPACE_PREFIX, workspace_template)
    os.chdir(WORKSPACE_PREFIX)
    workspace.certify()
    aggregator.generate_cert_request(agg_fqdn)
    aggregator.certify(agg_fqdn, silent=True)
    data_path = 1
    for col_name in col_names:
        collaborator.generate_cert_request(
            col_name, str(data_path), silent=True, skip_package=True)
        collaborator.certify(col_name, silent=True)
        data_path += 1

    setup_logging()


def create_collaborator(plan, name, model, aggregator):
    """
    Create the collaborator.

    Using the same plan object to create multiple collaborators leads to
    identical collaborator objects. This function can be removed once
    collaborator generation is fixed in openfl/federated/plan/plan.py
    """
    plan = copy(plan)

    return plan.get_collaborator(name, task_runner=model, client=aggregator)


def run_experiment(collaborator_dict, override_config={}):
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

    Returns:
        final_federated_model : FederatedModel
            The final model resulting from the federated learning experiment
    """
    from sys import path

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(str(root))
    path.insert(0, str(work))

    # Update the plan if necessary
    if len(override_config) > 0:
        update_plan(override_config)

    # TODO: Fix this implementation. The full plan parsing is reused here,
    # but the model and data will be overwritten based on user specifications
    plan_config = 'plan/plan.yaml'
    cols_config = 'plan/cols.yaml'
    data_config = 'plan/data.yaml'

    plan = Plan.Parse(plan_config_path=Path(plan_config),
                      cols_config_path=Path(cols_config),
                      data_config_path=Path(data_config))

    # Overwrite plan values
    plan.authorized_cols = list(collaborator_dict)
    tensor_pipe = plan.get_tensor_pipe()

    # This must be set to the final index of the list (this is the last
    # tensorflow session to get created)
    plan.runner_ = list(collaborator_dict.values())[-1]
    model = plan.runner_

    # Initialize model weights
    init_state_path = plan.config['aggregator']['settings']['init_state_path']
    rounds_to_train = plan.config['aggregator']['settings']['rounds_to_train']
    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
        logger,
        plan.runner_.get_tensor_dict(False)
    )

    model_snap = utils.construct_model_proto(tensor_dict=tensor_dict,
                                             round_number=0,
                                             tensor_pipe=tensor_pipe)

    logger.info(f'Creating Initial Weights File    🠆 {init_state_path}')

    utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

    logger.info('Starting Experiment...')

    aggregator = plan.get_aggregator()

    model_states = {
        collaborator: None for collaborator in collaborator_dict.keys()
    }

    # Create the collaborators
    collaborators = {
        collaborator: create_collaborator(
            plan, collaborator, model, aggregator
        ) for collaborator in plan.authorized_cols
    }

    for round_num in range(rounds_to_train):
        for col in plan.authorized_cols:

            collaborator = collaborators[col]
            model.set_data_loader(collaborator_dict[col].data_loader)

            if round_num != 0:
                model.rebuild_model(round_num, model_states[col])

            collaborator.run_simulation()

            model_states[col] = model.get_tensor_dict(with_opt_vars=True)

    # Set the weights for the final model
    model.rebuild_model(
        rounds_to_train - 1, aggregator.last_tensor_dict, validation=True)
    return model
