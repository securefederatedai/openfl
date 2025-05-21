# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AggregatorClientInterface module."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class AggregatorClientInterface(ABC):
    @abstractmethod
    def ping(self):
        """
        Ping the aggregator to check connectivity.
        """
        pass

    @abstractmethod
    def get_tasks(self) -> Tuple[List[Any], int, int, bool]:
        """
        Retrieves tasks for the given collaborator client.
        Returns a tuple: (tasks, round_number, sleep_time, time_to_quit)
        """
        pass

    @abstractmethod
    def get_aggregated_tensor(
        self,
        tensor_name: str,
        round_number: int,
        report: bool,
        tags: List[str],
        require_lossless: bool,
    ) -> Any:
        """
        Retrieves the aggregated tensor.
        """
        pass

    @abstractmethod
    def send_local_task_results(
        self,
        round_number: int,
        task_name: str,
        data_size: int,
        named_tensors: List[Any],
    ) -> Any:
        """
        Sends local task results.
        Parameters:
          collaborator_name: Name of the collaborator.
          round_number: The current round.
          task_name: Name of the task.
          data_size: Size of the data.
          named_tensors: A list of tensors (or named tensor objects).
        Returns a SendLocalTaskResultsResponse.
        """
        pass

    @abstractmethod
    def send_message_to_server(self, openfl_message: Any, collaborator_name: str) -> Any:
        """
        Forwards a converted message from the local client to the OpenFL server and returns the
        response.
        Args:
            openfl_message: The converted message to be sent to the OpenFL server (InteropMessage
                proto).
            collaborator_name: The name of the collaborator.
        Returns:
            The response from the OpenFL server (InteropMessage proto).
        """
        pass
