# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import zipfile
from typing import Union

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

    NOTE: if `prefer_minimal` is False, cwd must be the workspace directory
    because we need to construct dataloader from actual collaborator data path
    with actual data present.

    Args:
        plan (Plan):
            plan object linked with the dataloader
        prefer_minimal (bool ?):
            prefer to use MockDataLoader which can be used to more easily
            instantiate task_runner without any initial data.
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

    # if specified, try to use minimal dataloader
    if prefer_minimal:
        # if input_shape not given, try to ascertain input_shape from plan
        if not input_shape and "input_shape" in plan.config["data_loader"]["settings"]:
            input_shape = plan.config["data_loader"]["settings"]["input_shape"]

        # input_shape is resolved; we can use the minimal dataloader intended
        # for util contexts which does not need a full dataloader with data
        if input_shape:
            data_loader: DataLoader = MockDataLoader(input_shape)
            # generically inherit all attributes from data_loader.settings
            for key, value in plan.config["data_loader"]["settings"].items():
                setattr(data_loader, key, value)
            return data_loader

    # Fallback; try to get a dataloader by constructing it from the collaborator
    # data directory path present in the the current workspace

    collaborator_names = list(plan.cols_data_paths)
    collatorators_count = len(collaborator_names)

    if collaborator_index >= collatorators_count:
        raise Exception(
            f"Unable to construct full dataloader from collab_index={collaborator_index} "
            f"when the plan has {collatorators_count} as total collaborator count. "
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
