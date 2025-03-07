# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Collaborator module."""

import logging
from typing import List, Optional

from openfl.transport.grpc.aggregator_client import AggregatorGRPCClient
from openfl.utilities import TensorKey
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

    def __init__(
        self,
        collaborator_name,
        aggregator_uuid,
        federation_uuid,
        client: AggregatorGRPCClient,
        task_runner,
        task_config,
        opt_treatment="RESET",
        device_assignment_policy="CPU_ONLY",
        use_delta_updates=False,
        compression_pipeline=None,
        db_store_rounds=1,
        log_memory_usage=False,
        write_logs=False,
        callbacks: Optional[List] = [],
        secure_aggregation=False,
    ):
        super().__init__(
            collaborator_name,
            aggregator_uuid,
            federation_uuid,
            client,
            task_runner,
            task_config,
            opt_treatment,
            device_assignment_policy,
            use_delta_updates,
            compression_pipeline,
            db_store_rounds,
            log_memory_usage,
            write_logs,
            callbacks,
            secure_aggregation
        )

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
            # TODO: Need to determine a more general way to handle this in order to enable
            # additional tasks to be added to be added to Connector
            if hasattr(self.task_runner, func_name):
                method = getattr(self.task_runner, func_name)
                if callable(method):
                    framework = self.task_config['settings']["connect_to"]
                    LocalGRPCServer = connector.get_local_grpc_server(framework)
                    local_grpc_server = LocalGRPCServer(self.client, self.collaborator_name)
                    method(local_grpc_server, **kwargs) 
                    # TODO: better to use self.send_task_results(global_output_tensor_dict, round_number, task_name)
                    # maybe set global_output_tensor to empty
                    self.client.send_local_task_results(self.collaborator_name, round_number, task_name)
                    metrics = {f'{self.collaborator_name}/start_client_adapter': 'Completed'}
                    return metrics
                else:
                    raise AttributeError(f"{func_name} is not callable on {self.task_runner}")
            else:
                raise AttributeError(f"{func_name} does not exist on {self.task_runner}")