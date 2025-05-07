# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import importlib
import logging
import os
import zipfile

from openfl.federated import Plan
from openfl.federated.data.loader import DataLoader

logger = logging.getLogger(__name__)


def initialize_minimal_dataloader(plan: Plan) -> DataLoader:
    """Initialize a minimal dataloader without loading actual data.

    This is used primarily for model initialization when the actual data
    is not needed.

    Args:
        plan (Plan): Plan object linked with the dataloader

    Returns:
        DataLoader: A minimal dataloader instance with no data loaded

    Raises:
        ValueError: If required configuration is missing or dataloader class cannot be found
    """
    # Get the dataloader template from plan
    if "data_loader" not in plan.config or "template" not in plan.config["data_loader"]:
        logger.error("Missing 'data_loader' or 'template' field in plan configuration")
        raise ValueError("Invalid plan configuration: missing data_loader template")

    dataloader_template = plan.config["data_loader"]["template"]

    # Dynamically import the dataloader class
    module_name, class_name = dataloader_template.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        logger.error(f"Class {class_name} not found in module {module_name}")
        raise ValueError(
            f"Cannot load data_loader class '{class_name}' from module '{module_name}'"
        )

    dataloader_class = getattr(module, class_name)

    # Initialize dataloader with None as data_path to skip data loading
    if "settings" not in plan.config["data_loader"]:
        logger.error("Missing 'settings' field in data_loader configuration")
        raise ValueError("Invalid plan configuration: missing data_loader settings")

    data_loader_settings = plan.config["data_loader"]["settings"].copy()
    data_loader = dataloader_class(data_path=None, **data_loader_settings)
    logger.info("Initialized minimal dataloader for model creation")
    return data_loader


def initialize_dataloader(
    plan: Plan,
    collaborator_index: int = 0,
) -> DataLoader:
    """Initialize a dataloader with actual data from a collaborator.

    NOTE: cwd must be the workspace directory because we need to
    construct dataloader from actual collaborator data path
    with actual data present.

    Args:
        plan (Plan): Plan object linked with the dataloader
        collaborator_index (int): Which collaborator should be used for initializing dataloader
            among collaborators specified in plan/data.yaml. Defaults to 0.

    Returns:
        DataLoader: A dataloader instance with data loaded

    Raises:
        ValueError: If collaborator_index is out of range
    """
    # Regular dataloader initialization with actual data paths
    collaborator_names = list(plan.cols_data_paths)
    collaborators_count = len(collaborator_names)

    if collaborator_index >= collaborators_count:
        raise ValueError(
            f"Unable to construct full dataloader from collab_index={collaborator_index} "
            f"when the plan has {collaborators_count} as total collaborator count. "
            f"Please check plan/data.yaml file for current collaborator entries."
        )

    collaborator_name = collaborator_names[collaborator_index]
    collaborator_data_path = plan.cols_data_paths[collaborator_name]

    # use seed_data provided by data_loader config if available
    if "seed_data" in plan.config["data_loader"]["settings"] and not os.path.isdir(
        collaborator_data_path
    ):
        os.makedirs(collaborator_data_path)
        sample_data_zip_file = plan.config["data_loader"]["settings"]["seed_data"]
        with zipfile.ZipFile(sample_data_zip_file, "r") as zip_ref:
            zip_ref.extractall(collaborator_data_path)

    data_loader = plan.get_data_loader(collaborator_name)
    return data_loader
