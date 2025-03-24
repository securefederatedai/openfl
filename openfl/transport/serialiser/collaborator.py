# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
This file contains the CollaboratorSerialiser which is used as a middleware between the
collaborator component and aggregator client.
"""

from openfl.pipelines import TensorCodec


class CollaboratorSerialiser:
    """
    A class used to serialise and deserialise tensors for a collaborator in a federated learning
    setup.
    """

    def __init__(
        self,
        collaborator_name,
        client,
        compression_pipeline,
    ):
        self._collaborator_name = collaborator_name
        self._aggregator_client = client
        self._tensor_codec = TensorCodec(compression_pipeline)

    def get_tasks(self):
        """
        Retrieves the tasks assigned to the collaborator from the aggregator client.
        """
        return self._aggregator_client.get_tasks()

    def get_aggregated_tensor(
        self,
        tensor_name: str,
        round_number: int,
        report: bool,
        tags: tuple,
        require_lossless: bool,
    ):
        """
        Retrieves and deserializes an aggregated tensor from the aggregator.

        Args:
            tensor_name (str): The name of the tensor to retrieve.
            round_number (int): The round number associated with the tensor.
            report (bool): Whether to report the retrieval process.
            tags (tuple): Tags associated with the tensor.
            require_lossless (bool): Whether lossless retrieval is required.

        Returns:
            tuple: A tuple containing the tensor key and the deserialized numpy array.
        """
        tensor = self._aggregator_client.get_aggregated_tensor(
            tensor_name,
            round_number,
            report,
            tags,
            require_lossless,
        )
        tensor_key, nparray = self._tensor_codec.deserialise(tensor, self._collaborator_name)

        return tensor_key, nparray

    def send_local_task_results(
        self,
        round_number: int,
        task_name: str,
        data_size: int = None,
        tensor_dict: dict = {},
    ):
        """
        Sends the local task results to the aggregator client.

        Args:
            round_number (int): The current round number of the task.
            task_name (str): The name of the task.
            data_size (int, optional): The size of the data. Defaults to None.
            tensor_dict (dict, optional): A dictionary where keys are tensor names and values
                are numpy arrays. Defaults to {}.

        Returns:
            None
        """
        named_tensors = [
            self._tensor_codec.serialise(tensor_key, nparray)
            for tensor_key, nparray in tensor_dict.items()
        ]
        self._aggregator_client.send_local_task_results(
            round_number,
            task_name,
            data_size,
            named_tensors,
        )
