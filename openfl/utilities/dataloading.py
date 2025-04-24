# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import zipfile
from typing import Optional, Union

from openfl.federated import Plan
from openfl.federated.data.loader import DataLoader
from openfl.utilities.mocks import MockDataLoader


def get_dataloader(
    plan: Plan,
    prefer_minimal: bool = False,
    input_shape: Union[list, dict] = None,
    collaborator_index: int = 0,
) -> DataLoader:
    """Get dataloader instance from plan

    Args:
        plan (Plan):
            plan object linked with the dataloader
        prefer_minimal (bool ?):
            prefer to use MockDataLoader which can be used to more easily
            instantiate task_runner without any initial data.
            This is typically used when running on the model owner/aggregator.
            Default to `False`.
        input_shape (list | dict ?):
            input_shape denoted by list notation `[a,b,c, ...]` or in case
            of multihead models, dict object with individual layer keys such
            as `{"input_0": [a,b,...], "output_1": [x,y,z, ...]}`
            Defaults to `None`.
        collaborator_index (int ?):
            which collaborator should be used for initializing dataloader
            among collaborators specified in plan/data.yaml.
            Defaults to `0`.

    Returns:
        data_loader (DataLoader): DataLoader instance
    """
    # Model owner initialization path - used during fx plan initialize and fx model save
    if prefer_minimal:
        return _get_minimal_dataloader(plan, input_shape)
    
    # Collaborator path - used when actually running the federation
    else:
        return _get_collaborator_dataloader(plan, collaborator_index)


def _get_minimal_dataloader(
    plan: Plan, 
    input_shape: Optional[Union[list, dict]] = None
) -> DataLoader:
    """Get a minimal dataloader for model initialization on the model owner/aggregator.
    This doesn't require actual data to be present and won't attempt to validate data paths.

    Args:
        plan: The plan object
        input_shape: Input shape specification

    Returns:
        DataLoader: A minimal dataloader suitable for model initialization
    """
    # Try to get input_shape from plan if not provided explicitly
    if not input_shape and "input_shape" in plan.config["data_loader"]["settings"]:
        input_shape = plan.config["data_loader"]["settings"]["input_shape"]

    # Create a mock dataloader with the input shape
    data_loader: DataLoader = MockDataLoader(input_shape)
    
    # Inherit all attributes from data_loader.settings except input_shape
    # to avoid overriding the explicitly provided shape
    for key, value in plan.config["data_loader"]["settings"].items():
        if key != "input_shape":  # Skip input_shape to preserve the one used for initialization
            setattr(data_loader, key, value)
    
    return data_loader


def _get_collaborator_dataloader(plan: Plan, collaborator_index: int = 0) -> DataLoader:
    """Get a dataloader for an actual collaborator with real data.

    This will check for data path existence and handle seed data if provided.

    Args:
        plan: The plan object
        collaborator_index: Which collaborator's data to use

    Returns:
        DataLoader: A dataloader configured for the specified collaborator
    """
    collaborator_names = list(plan.cols_data_paths)
    collaborator_count = len(collaborator_names)

    if collaborator_index >= collaborator_count:
        raise Exception(
            f"Unable to construct dataloader for index={collaborator_index} "
            f"when the plan has {collaborator_count} total collaborators. "
            f"Please check plan/data.yaml file for current collaborator entries."
        )

    collaborator_name = collaborator_names[collaborator_index]
    data_path = plan.cols_data_paths[collaborator_name]

    # Handle seed data if provided in the plan
    if "seed_data" in plan.config["data_loader"]["settings"]:
        seed_data_zip = plan.config["data_loader"]["settings"]["seed_data"]

        if os.path.isfile(seed_data_zip):
            os.makedirs(data_path, exist_ok=True)

            # Always extract seed data when it's provided
            # This ensures fresh data even if the directory already exists
            with zipfile.ZipFile(seed_data_zip, "r") as zip_ref:
                zip_ref.extractall(data_path)
        else:
            logging.getLogger(__name__).warning(
                f"Seed data specified ({seed_data_zip}) but file not found"
            )

    # Get the actual dataloader from the plan
    data_loader = plan.get_data_loader(collaborator_name)
    return data_loader
