# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from metaflow import Flow
import logging
import numpy as np
from openfl.databases import TensorDB
from openfl.utilities import TensorKey

import tests.end_to_end.utils.exceptions as ex

log = logging.getLogger(__name__)


def validate_flow(flow_obj, expected_flow_steps):
    """
    Validate:
    1. If the given training round were completed
    2. If all the steps were executed
    3. If each collaborator step was executed
    4. If end was executed once
    """

    cli_flow_obj = Flow("TestFlowInternalLoop")  # Flow object from CLI
    cli_flow_steps = list(cli_flow_obj.latest_run)  # Steps from CLI
    cli_step_names = [step.id for step in cli_flow_steps]

    # 1. If the given training round were completed
    assert flow_obj.training_rounds == flow_obj.train_count, "Number of training completed is not equal to training rounds"

    for step in cli_flow_steps:
        task_count = 0
        func = getattr(flow_obj, step.id)
        for task in list(step):
            task_count = task_count + 1

        # Each aggregator step should be executed for training rounds times
        if (
            (func.aggregator_step is True)
            and (task_count != flow_obj.training_rounds)
            and (step.id != "end")
        ):
            assert False, f"More than one execution detected for Aggregator Step: {step}"

        # Each collaborator step is executed for (training rounds)*(number of collaborator) times
        if (func.collaborator_step is True) and (
            task_count != len(flow_obj.collaborators) * flow_obj.training_rounds
        ):
            assert False, f"Incorrect number of execution detected for Collaborator Step: {step}. Expected: {flow_obj.training_rounds*len(flow_obj.collaborators)} Actual: {task_count}"

    steps_present_in_cli = [
        step for step in expected_flow_steps if step in cli_step_names
    ]
    missing_steps_in_cli = [
        step for step in expected_flow_steps if step not in cli_step_names
    ]
    extra_steps_in_cli = [
        step for step in cli_step_names if step not in expected_flow_steps
    ]
    return steps_present_in_cli, missing_steps_in_cli, extra_steps_in_cli


def init_collaborator_private_attr_index(param):
        """
        Initialize a collaborator's private attribute index.

        Args:
            param (int): The initial value for the index.

        Returns:
            dict: A dictionary with the key 'index' and the value of `param` incremented by 1.
        """
        return {"index": param + 1}


def init_collaborator_private_attr_name(param):
        """
        Initialize a collaborator's private attribute name.

        Args:
            param (str): The name to be assigned to the collaborator's private attribute.

        Returns:
            dict: A dictionary with the key 'name' and the value of the provided parameter.
        """
        return {"name": param}


def init_collaborate_pvt_attr_np(param):
    """
    Initialize private attributes for collaboration with numpy arrays.

    This function generates random numpy arrays for training and testing loaders
    based on the given parameter.

    Args:
        param (int): A multiplier to determine the size of the generated arrays.

    Returns:
        dict: A dictionary containing:
            - "train_loader" (numpy.ndarray): A numpy array of shape (param * 50, 28, 28) with random values.
            - "test_loader" (numpy.ndarray): A numpy array of shape (param * 10, 28, 28) with random values.
    """
    return {
        "train_loader": np.random.rand(param * 50, 28, 28),
        "test_loader": np.random.rand(param * 10, 28, 28),
    }


def init_agg_pvt_attr_np():
    """
    Initialize a dictionary with a private attribute for testing.

    Returns:
        dict: A dictionary containing a single key "test_loader" with a value
              of a NumPy array of shape (10, 28, 28) filled with random values.
    """
    return {"test_loader": np.random.rand(10, 28, 28)}


def callable_to_init_collab_unserializable_pvt_attrs():
    """
    Create and return a TensorDB
    """
    return {"col_tensor_db": TensorDB()}


def callable_to_init_agg_unserializable_pvt_attrs():
    """
    Create and return a TensorDB
    """
    return {"agg_tensor_db": TensorDB()}


def run_notebook(notebook_path, output_notebook_path):
    """
    Function to run the notebook.
    Args:
        notebook_path (str): Path to the notebook
        participant_res_files (dict): Dictionary containing participant names and their result log files
    Returns:
        bool: True if successful, else False
    """
    import papermill as pm
    try:
        log.info(f"Running the notebook: {notebook_path} with output notebook path: {output_notebook_path}")
        output = pm.execute_notebook(
            input_path=notebook_path,
            output_path=output_notebook_path,
            request_save_on_cell_execute=True,
            autosave_cell_every=5, # autosave every 5 seconds
            log_output=True,
        )
    except pm.exceptions.PapermillExecutionError as e:
        log.error(f"PapermillExecutionError: {e}")
        raise e

    except ex.NotebookRunException as e:
        log.error(f"Failed to run the notebook: {e}")
        raise e
    return True
