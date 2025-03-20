# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Collaborator module."""

import logging
from src.grpc import connector

from openfl.component import Collaborator

logger = logging.getLogger(__name__)

class CollaboratorFlower(Collaborator):
    r"""The Collaborator object class.

    Attributes:
        collaborator_name (str): The common name for the collaborator.
        aggregator_uuid (str): The unique id for the client.
        federation_uuid (str): The unique id for the federation.
        client (object): The client object.
        task_runner (object): The task runner object.
        task_config (dict): The task configuration.
        opt_treatment (str)*: The optimizer state treatment.
        device_assignment_policy (str): [Deprecated] The device assignment policy.
        use_delta_updates (bool)*: If True, only model delta gets sent. If False,
            whole model gets sent to collaborator.
        compression_pipeline (object): The compression pipeline.
        db_store_rounds (int): The number of rounds to store in the database.
        single_col_cert_common_name (str): The common name for the single
            column certificate.

    .. note::
        \* - Plan setting.
    """

    def do_task(self, task, round_number) -> dict:
        """Perform the specified task.

        Args:
            task (list_of_str): List of tasks.
            round_number (int): Actual round number.

        Returns:
            A dictionary of reportable metrics of the current collaborator for the task.
        """
        # map this task to an actual function name and kwargs
        if isinstance(task, str):
            task_name = task
        else:
            task_name = task.name
        func_name = self.task_config[task_name]["function"]
        kwargs = self.task_config[task_name]["kwargs"]
        if func_name=="start_client_adapter":
            if hasattr(self.task_runner, func_name):
                method = getattr(self.task_runner, func_name)
                if callable(method):
                    framework = self.task_config['settings']["connect_to"]
                    LocalGRPCServer = connector.get_interop_server(framework)
                    interop_server = LocalGRPCServer(self.client, self.collaborator_name)
                    method(interop_server, **kwargs)
                    self.client.send_local_task_results(round_number, task_name)
                    metrics = {f'{self.collaborator_name}/start_client_adapter': 'Completed'}
                    return metrics
                else:
                    raise AttributeError(f"{func_name} is not callable on {self.task_runner}")
            else:
                raise AttributeError(f"{func_name} does not exist on {self.task_runner}")
