# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
This file contains the Setup class used on the server side for secure
aggregation setup.
"""

import json
import logging
import struct
from importlib import util

from openfl.utilities import TensorKey

# Check if pycryptodome is installed.
if util.find_spec("Crypto") is None:
    raise Exception(
        "'pycryptodome' not installed.This package is necessary when secure aggregation is enabled."
    )

# Import packages, if installed.
from openfl.utilities.secagg import (
    calculate_shared_mask,
    generate_agreed_key,
    reconstruct_secret,
)

logger = logging.getLogger(__name__)


class SecAggSetup:
    """
    Used by the aggregator for the setup stage of secure aggregation.
    """

    def __init__(self, aggregator_uuid, collaborator_list, tensor_db):
        self._aggregator_uuid = aggregator_uuid
        self._collaborator_list = collaborator_list
        self._tensor_db = tensor_db
        self._results = {}

    def process_secagg_setup_tensors(self, named_tensors) -> bool:
        """
        Set up secure aggregation for the given collaborator and named tensors.

        This method processes named tensors that are part of the secure
        aggregation setup stages. It saves the processed tensors to the local
        tensor database and checks if all collaborators have sent their data
        for the current key. If all collaborators have sent their data, it
        proceeds with aggregation for the key.

        Args:
            named_tensors (list): A list of named tensors to be processed.

        Returns:
            bool: True if the received tensors belong to secagg setup stage,
                False otherwise.
        """
        secagg_setup = False
        for named_tensor in named_tensors:
            # Check if the tensor belongs to one from secure aggregation
            # setup stages.
            if "secagg" in tuple(named_tensor.tags):
                secagg_setup = True
                # Process and save tensor to local tensor db.
                self._save_secagg_tensor(named_tensor)
                tensor_name = named_tensor.name
                # Check if all collaborators have sent their data for the
                # current key.
                all_collaborators_sent = self._check_tensors_received(tensor_name)
                if not all_collaborators_sent:
                    continue
                # If all collaborators have sent their data, proceed with
                # aggregation for the key.
                self._aggregate_tensor(tensor_name)

        return secagg_setup

    def _check_tensors_received(self, tensor_name):
        """
        Checks if the tensor with the given name has been received from all
        collaborators.

        Args:
            tensor_name (str): The name of the tensor to check.

        Returns:
            bool: True if the tensor has been received from all collaborators,
                False otherwise.
        """
        logger.debug("Checking if received {} from all collaborators".format(tensor_name))
        all_received = True
        for collaborator in self._collaborator_list:
            nparray = self._tensor_db.get_tensor_from_cache(
                TensorKey(
                    tensor_name,
                    self._aggregator_uuid,
                    -1,
                    False,
                    (
                        collaborator,
                        "secagg",
                    ),
                )
            )
            if nparray is None:
                all_received = False

        return all_received

    def _aggregate_tensor(self, tensor_name):
        """
        Aggregates the specified tensor based on its name and performs
        subsequent operations if necessary.

        Args:
            tensor_name (str): The name of the tensor to aggregate.
                It can be one of the following: "public_key", "ciphertext",
                "seed_share", "key_share".

        Raises:
            ValueError: If the tensor_name is not one of the expected values.

        Operations:
            - Aggregates public keys if tensor_name is "public_key".
            - Aggregates ciphertexts if tensor_name is "ciphertext".
            - Aggregates seed shares if tensor_name is "seed_share".
            - Aggregates key shares if tensor_name is "key_share".
            - If both "seed_shares" and "key_shares" are present in the
                results, it:
                - Reconstructs secrets.
                - Generates agreed keys between all pairs of collaborators.
                - Saves the local tensors to the tensor database.
        """
        if tensor_name == "public_key":
            self._aggregate_public_keys()
        elif tensor_name == "ciphertext":
            self._aggregate_ciphertexts()
        elif tensor_name in ["seed_share", "key_share"]:
            self._aggregate_secret_shares(tensor_name)

        if "seed_shares" in self._results and "key_shares" in self._results:
            self._reconstruct_secrets()
            # Generate agreed keys between all pairs of collaborators.
            self._generate_agreed_keys()
            # Save the local tensors to the tensor database.
            self._save_tensors()

    def _aggregate_public_keys(self):
        """
        Sorts the public keys received from collaborators and updates the
        results.
        """
        aggregated_tensor = []
        self._results["public_keys"] = {}
        self._results["index"] = {}
        index = 1
        for collaborator in self._collaborator_list:
            # Fetching public key for each collaborator from tensor db.
            nparray = self._tensor_db.get_tensor_from_cache(
                TensorKey(
                    "public_key",
                    self._aggregator_uuid,
                    -1,
                    False,
                    (
                        collaborator,
                        "secagg",
                    ),
                )
            )
            aggregated_tensor.append([index, nparray[0], nparray[1]])
            # Creating a map for local use.
            self._results["public_keys"][index] = [nparray[0], nparray[1]]
            self._results["index"][collaborator] = index
            index += 1

        # Storing the aggregated result in tensor db which is fetched by the
        # collaborators in subsequent steps.
        self._tensor_db.cache_tensor(
            {
                TensorKey(
                    "public_keys", self._aggregator_uuid, -1, False, ("secagg",)
                ): aggregated_tensor
            }
        )

    def _aggregate_ciphertexts(self):
        """
        Sorts the ciphertexts received from collaborators and updates the
        results.
        """
        aggregated_tensor = []
        self._results["ciphertexts"] = []

        for collaborator in self._collaborator_list:
            # Fetching ciphertext for each collaborator from tensor db.
            nparray = self._tensor_db.get_tensor_from_cache(
                TensorKey(
                    "ciphertext",
                    self._aggregator_uuid,
                    -1,
                    False,
                    (
                        collaborator,
                        "secagg",
                    ),
                )
            )
            for ciphertext in nparray:
                aggregated_tensor.append(ciphertext)
                # Creating a map for local use.
                self._results["ciphertexts"].append(ciphertext)
        # Storing the aggregated result in tensor db which is fetched by the
        # collaborators in subsequent steps.
        self._tensor_db.cache_tensor(
            {
                TensorKey(
                    "ciphertexts", self._aggregator_uuid, -1, False, ("secagg",)
                ): aggregated_tensor
            }
        )

    def _aggregate_secret_shares(self, key_name):
        """
        Aggregates secret shares for a given key name from the tensor database.

        This method fetches seed shares for each collaborator from the tensor
        database and organizes them into a dictionary for local use.

        Args:
            key_name (str): The name of the key for which secret shares are to
                be aggregated.
        """
        self._results[f"{key_name}s"] = {}

        for collaborator in self._collaborator_list:
            # Fetching seed shares for each collaborator from tensor db.
            nparray = self._tensor_db.get_tensor_from_cache(
                TensorKey(
                    key_name,
                    self._aggregator_uuid,
                    -1,
                    False,
                    (
                        collaborator,
                        "secagg",
                    ),
                )
            )
            for share in nparray:
                # Creating a map for local use.
                if int(share[1]) not in self._results[f"{key_name}s"]:
                    self._results[f"{key_name}s"][int(share[1])] = {}
                self._results[f"{key_name}s"][int(share[1])][int(share[0])] = share[2][2:-1]

    def _reconstruct_secrets(self):
        """
        Reconstructs the private seeds and private keys from the secret shares.
        """
        self._results["private_seeds"] = {}
        self._results["private_keys"] = {}

        for source_id in self._results["seed_shares"]:
            self._results["private_seeds"][source_id] = struct.unpack(
                "d", reconstruct_secret(self._results["seed_shares"][source_id])
            )[0]
            self._results["private_keys"][source_id] = reconstruct_secret(
                self._results["key_shares"][source_id]
            )
        logger.info("SecAgg: recreated secrets successfully")

    def _generate_agreed_keys(self):
        """
        Generates agreed keys between all pairs of collaborators using their
        private keys and public keys.
        """
        self._results["agreed_keys"] = []
        for source_index in self._results["index"].values():
            for dest_index in self._results["index"].values():
                if source_index == dest_index:
                    continue
                self._results["agreed_keys"].append(
                    [
                        source_index,
                        dest_index,
                        generate_agreed_key(
                            self._results["private_keys"][source_index],
                            self._results["public_keys"][dest_index][0],
                        ),
                    ]
                )

    def _save_tensors(self):
        """
        Generate and save tensors required for secure aggregation.

        This method generates private and shared masks by calling the
        `_generate_masks` method. It then creates a dictionary of tensors
        to be saved, which includes the sum of private and shared masks.
        The tensors are cached in the tensor database.

        These tensors are then added to the gradient before to get the
        actual aggregate after removing the masks.
        """
        shared_mask_sum = calculate_shared_mask(self._results["agreed_keys"])
        local_tensor_dict = {
            TensorKey("indices", "agg", -1, False, ("secagg",)): [
                [collaborator, index] for collaborator, index in self._results["index"].items()
            ],
            TensorKey("private_seeds", "agg", -1, False, ("secagg",)): [
                [index, seed] for index, seed in self._results["private_seeds"].items()
            ],
            TensorKey("agreed_keys", "agg", -1, False, ("secagg",)): self._results["agreed_keys"],
            TensorKey("shared_mask_sum", "agg", -1, False, ("secagg",)): [shared_mask_sum],
        }
        self._tensor_db.cache_tensor(local_tensor_dict)
        logger.info("SecAgg: setup completed, saved required tensors to db.")

    def _save_secagg_tensor(self, named_tensor):
        """
        Converts secure aggregation setup related named tensor to nparray
        and saves them to tensordb.
        """
        # The tensor has already been transfered to aggregator,
        # so the newly constructed tensor should have the aggregator origin
        tensor_key = TensorKey(
            named_tensor.name,
            self._aggregator_uuid,
            named_tensor.round_number,
            named_tensor.report,
            tuple(named_tensor.tags),
        )
        _, _, _, _, tags = tensor_key
        # Secure aggregation setup stage key
        if "secagg" in tags:
            nparray = json.loads(named_tensor.data_bytes)
            self._tensor_db.cache_tensor({tensor_key: nparray})
            logger.debug("Created TensorKey: %s", tensor_key)
