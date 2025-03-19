# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Federated averaging with secure aggregation module."""

import numpy as np

from openfl.interface.aggregation_functions.weighted_average import WeightedAverage
from openfl.utilities import LocalTensor


class SecureWeightedAverage(WeightedAverage):
    """Weighted average with secure aggregation."""

    def __init__(self):
        super().__init__()
        self._private_masks, self._shared_masks = None, None
        self.private_seeds, self.agreed_keys, self.col_indices = None, None, None

    def call(self, local_tensors, db_iterator, *_) -> np.ndarray:
        """Aggregate tensors.

        Args:
            local_tensors (list[openfl.utilities.LocalTensor]): List of local
                tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight','fc2.bias'.
                - 'round': 0-based number of round corresponding to this
                    tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training
                        result.
                        These tensors are passed to the aggregator node after
                        local learning.
                    - 'aggregated' indicates that tensor is a result of
                        aggregation.
                        These tensors are sent to collaborators for the next
                        round.
                    - 'delta' indicates that value is a difference between
                        rounds for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            np.ndarray: aggregated tensor
        """
        # Generate masks for the collaborators if not already done.
        self._generate_masks(db_iterator)
        # Calaculate the weighted avreage of collaborator masks.
        weighted_mask = self._calculcate_weighted_mask_average(self._private_masks, local_tensors)
        # Get weighted average for shared tensors.
        tensor_avg = super().call(local_tensors)
        # Subtract weighted average of masks from the tensor average.
        return np.subtract(np.subtract(tensor_avg, weighted_mask), self._shared_masks)

    def _generate_masks(self, db_iterator):
        """
        Generate shared and private masks for secure aggregation.

        This method processes a database iterator to extract private seeds,
        agreed keys, and column indices, which are then used to generate
        shared and private masks.

        Args:
            db_iterator (iterator): An iterator over the database items
                containing tensors with tags, tensor names, and numpy arrays.

        Raises:
            KeyError: If the required keys are not found in the database items.

        Notes:
            - The shared masks are calculated using the agreed keys.
            - The private masks are generated for each collaborator using
                their private seeds.
            - The private masks are stored in a dictionary with the
                collaborator's name as the key.
        """
        from openfl.utilities.secagg import (
            calculate_shared_mask,
            pseudo_random_generator,
        )

        if self._shared_masks and self._private_masks:
            return

        # Get all required values from tensor db.
        self._get_secagg_items_from_db(db_iterator)
        if not self._shared_masks:
            # Calculate shared mask
            self._shared_masks = calculate_shared_mask(self.agreed_keys)

        if not self._private_masks:
            # Create a dict with collaborator index and their name.
            # This dict is used to map private masks to the collaborator name
            # as they are stored with collaborator index in the db.
            col_idx = {}
            for col in self.col_indices:
                col_idx[col[1]] = col[0]

            # Generate private masks for each collaborator.
            self._private_masks = {}
            for seed in self.private_seeds:
                # col_name: col_private_mask
                self._private_masks[col_idx[seed[0]]] = pseudo_random_generator(seed[1])

            del col_idx

    def _calculcate_weighted_mask_average(
        self, private_masks: dict, local_tensors: list[LocalTensor]
    ):
        """
        Calculate the weighted mask average for the given local tensors using
        their private masks and weight for their respective tensors.

        Args:
            private_masks (dict): A dictionary where keys are collaborator
                names and values are tuples, with the second element being the
                mask for that colaborator.
            local_tensors (list): A list of tensors, where each tensor has
                attributes 'col_name' and 'weight'.

        Returns:
            numpy.ndarray: The average mask calculated as the weighted
                average of the masks.
        """
        weights = []
        masks = []
        # Create a list of private masks and weights for the collaborators
        # whose tensors are being aggregated.
        for tensor in local_tensors:
            col_name = tensor.col_name
            weights.append(tensor.weight)
            masks.append(private_masks[col_name])

        # Calculate weighted mask using the masks and weights where each index
        # in the lists represents a single collaborator.
        weighted_mask = np.average(masks, weights=weights, axis=0)

        del weights
        del masks

        return weighted_mask

    def _get_secagg_items_from_db(self, db_iterator):
        """
        Extracts secure aggregation items from a database iterator.
        It retrieves the private seeds, agreed keys, and column indices from
        the database items.

        Args:
            db_iterator (iterable): An iterator that yields database items.
                Each item is expected to be a dictionary with keys "tags",
                "tensor_name", and "nparray".

        Raises:
            KeyError: If any of the required keys ("tags", "tensor_name",
                "nparray") are missing in an item.
        """
        for item in db_iterator:
            if "tags" in item and item["tags"] == ("secagg",):
                if item["tensor_name"] == "private_seeds":
                    self.private_seeds = item["nparray"]
                elif item["tensor_name"] == "agreed_keys":
                    self.agreed_keys = item["nparray"]
                elif item["tensor_name"] == "indices":
                    self.col_indices = item["nparray"]
