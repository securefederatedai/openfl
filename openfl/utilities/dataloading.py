# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import importlib
import logging
import os
import zipfile
from typing import Union

from openfl.federated import Plan
from openfl.federated.data.loader import DataLoader

logger = logging.getLogger(__name__)

def get_dataloader(
    plan: Plan,
    prefer_minimal: bool = False,
    input_shape: Union[list, dict] = None,
    collaborator_index: int = 0,
) -> DataLoader:
    """Get dataloader instance from plan

    NOTE: if `prefer_minimal` is False, cwd must be the workspace directory
    because we need to construct dataloader from actual collaborator data path
    with actual data present.

    Args:
        plan (Plan):
            plan object linked with the dataloader
        prefer_minimal (bool ?):
            prefer to initialize dataloader without loading actual data.
            Used primarily for model initialization.
            Default to `False`.
        input_shape (list | dict ?):
            Legacy parameter, now deprecated and will be ignored.
            Defaults to `None`.
        collaborator_index (int ?):
            which collaborator should be used for initializing dataloader
            among collaborators specified in plan/data.yaml.
            Defaults to `0`.

    Returns:
        data_loader (DataLoader): DataLoader instance
    """

    # If prefer_minimal is True, we attempt to create the dataloader without actual data
    if prefer_minimal:
        try:
            # Get the dataloader template from plan
            dataloader_template = plan.config["data_loader"]["template"]
            # Dynamically import the dataloader class
            module_name, class_name = dataloader_template.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                dataloader_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to import dataloader class: {e}")
                raise ValueError(
                    f"Cannot load data_loader class from template "
                    f"'{dataloader_template}'"
                ) from e

            # Initialize dataloader with None as data_path to skip data loading
            data_loader_settings = plan.config["data_loader"]["settings"].copy()
            data_loader = dataloader_class(data_path=None, **data_loader_settings)
            logger.info("Initialized minimal dataloader for model creation")
            return data_loader
        except KeyError:
            logger.error("Missing 'data_loader' or 'template' field in plan configuration")
            raise ValueError("Invalid plan configuration: missing data_loader template")
        except Exception as e:
            logger.warning(f"Could not initialize minimal dataloader: {e}")
            raise

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
