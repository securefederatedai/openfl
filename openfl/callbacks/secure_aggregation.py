# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
This file contains callback that help setup for secure aggregation for the
collaborator.
"""

import json
import logging
import struct

import numpy as np

from openfl.callbacks.callback import Callback
from openfl.protocols import utils
from openfl.utilities import TensorKey

logger = logging.getLogger(__name__)


class SecAggBootstrapping(Callback):
    """
    This callback is used by the collaborator to perform secure aggregation
    bootstrapping.

    Required params include:
    - origin: Name of the collaborator using the callback.
    - client: AggregatorGRPCClient to communicate with the aggregator server.

    It also requires the tensor-db client to be set.
    """

    def __init__(self):
        super().__init__()

        from importlib import util

        # Check if pycryptodome is installed.
        if util.find_spec("Crypto") is None:
            raise Exception(
                "'pycryptodome' not installed."
                "This package is necessary when secure aggregation is enabled."
            )

    def on_experiment_begin(self, logs=None):
        """
        Used to perform secure aggregation setup before experiment begins.
        """
        self.name = self.params["origin"]
        self.client = self.params["client"]
        logger.info("Secure aggregation is enabled, starting setup...")
        # Generate private public key pair used for secure aggregation.
        self._generate_keys()
        # Fetch public keys for all collaborators from the aggregator.
        collaborator_keys = self._fetch_public_keys()
        # Generate ciphertexts for each collaborator and share them with the
        # aggregator.
        self._generate_ciphertexts(collaborator_keys)
        # Decrypt the addressed ciphertexts and share them with the
        # aggregator.
        self._decrypt_ciphertexts(collaborator_keys)
        # Save the tensors which are required for masking of gradients.
        self._save_mask_tensors()

    def _generate_keys(self):
        """
        Generates a pair of private and public keys, along with a private seed,
        and updates the local and global results.

        This method performs the following steps:
        1. Generates two pairs of private and public keys.
        2. Creates a local result dictionary containing the private keys,
            public keys, and a private seed.
        3. Creates a global result dictionary containing the public keys.
        4. Sends the global results to the participant via the aggregator's
            secure aggregation mechanism.
        5. Updates the instance parameters with the local result.
        """
        from openfl.utilities.secagg import generate_key_pair

        private_key1, public_key1 = generate_key_pair()
        private_key2, public_key2 = generate_key_pair()

        local_result = {
            "private_key": [private_key1, private_key2],
            "public_key": [public_key1, public_key2],
            "private_seed": np.random.random(),
        }
        global_results = {
            "public_key": [public_key1, public_key2],
        }

        self._send_to_aggregator(global_results, "generate_keys")
        # Update callback params as the results for this step are reused at a
        # later stage.
        self.params.update(local_result)
        logger.debug("SecAgg: Generate key-pair generation successful")

    def _fetch_public_keys(self):
        """
        Fetches collaborators' public keys from the aggregator and identifies
        the index of the current collaborator using it's public key.

        Returns:
            dict: A dictionary containing the public keys of all collaborators,
                where the keys are the collaborator indices and the values are
                the public keys.
        """
        public_keys = {}
        public_keys_tensor = self._fetch_from_aggregator("public_keys")
        for tensor in public_keys_tensor:
            # Creating a dictionary of the received public keys.
            public_keys[int(tensor[0])] = [tensor[1], tensor[2]]
            # Finding the index of the current collaborator by matching the
            # first public key.
            if tensor[1] == self.params["public_key"][0]:
                self.index = int(tensor[0])

        return public_keys

    def _generate_ciphertexts(self, public_keys):
        """
        Generate ciphertexts for secure aggregation.

        This method generates ciphertexts for each collaborator using their
        public keys. It creates secret shares for the private seed and private
        key, then uses these shares to generate agreed keys and ciphertexts
        for secure communication between collaborators.

        Args:
            public_keys (dict): A dictionary where keys are collaborator
                indices and values are lists containing public keys of the
                collaborators.
        """
        from openfl.utilities.secagg import (
            create_ciphertext,
            create_secret_shares,
            generate_agreed_key,
        )

        logger.debug("SecAgg: Generating ciphertexts to be shared with other collaborators")
        collaborator_count = len(public_keys)

        private_seed = self.params["private_seed"]
        seed_shares = create_secret_shares(
            # Converts the floating-point number private_seed into an 8-byte
            # binary representation.
            struct.pack("d", private_seed),
            collaborator_count,
            collaborator_count,
        )

        private_keys = self.params["private_key"]
        # Create secret shares for the private key.
        key_shares = create_secret_shares(
            str.encode(private_keys[0]),
            collaborator_count,
            collaborator_count,
        )

        global_results = {"ciphertext": []}
        local_result = {"ciphertext_verification": {}, "agreed_keys": []}
        # Create cipher-texts for each collaborator.
        for collab_index in public_keys:
            agreed_key = generate_agreed_key(private_keys[0], public_keys[collab_index][0])
            ciphertext, mac, nonce = create_ciphertext(
                agreed_key,  # agreed key
                self.index,  # source collaborator index
                collab_index,  # destination collaborator index
                seed_shares[collab_index],  # seed share from source to dest
                key_shares[collab_index],  # key share from source to dest
            )
            global_results["ciphertext"].append((self.index, collab_index, str(ciphertext)))
            local_result["ciphertext_verification"][collab_index] = [ciphertext, mac, nonce]
            local_result["agreed_keys"].append([self.index, collab_index, agreed_key])

        self._send_to_aggregator(global_results, "generate_ciphertexts")
        # Update callback params as the results for this step are reused at a
        # later stage.
        self.params.update(local_result)

        logger.debug("SecAgg: Ciphertexts shared with the aggregator successfully")

    def _decrypt_ciphertexts(self, public_keys):
        """
        Decrypts the ciphertexts received from collaborators using the provided
        public keys.

        This method fetches the ciphertexts from the aggregator, decrypts them
        using the collaborator's private key and the provided public keys, and
        then sends the decrypted seed shares and key shares back to the
        aggregator.

        Args:
            public_keys (dict): A dictionary containing the public keys of the
                collaborators.
        """
        from openfl.utilities.secagg import (
            decipher_ciphertext,
            generate_agreed_key,
        )

        logger.debug("SecAgg: fetching addressed ciphertexts from the aggregator")

        ciphertexts = self._fetch_from_aggregator("ciphertexts")
        private_keys = self.params["private_key"]
        ciphertext_verification = self.params["ciphertext_verification"]

        global_results = {"seed_share": [], "key_share": []}

        for cipher in ciphertexts:
            source_index = int(cipher[0])
            if int(cipher[1]) == self.index:
                _, _, seed_share, key_share = decipher_ciphertext(
                    generate_agreed_key(private_keys[0], public_keys[source_index][0]),
                    ciphertext_verification[source_index][0],
                    ciphertext_verification[source_index][1],
                    ciphertext_verification[source_index][2],
                )
                global_results["seed_share"].append((source_index, self.index, str(seed_share)))
                global_results["key_share"].append((source_index, self.index, str(key_share)))

        self._send_to_aggregator(global_results, "decrypt_ciphertexts")

        logger.debug("SecAgg: decrypted ciphertexts shared with the aggregator")

    def _generate_masks(self):
        """
        Use the private seed and agreed keys to calculate the masks to be
        added to the gradients.
        """
        from openfl.utilities.secagg import (
            calculate_shared_mask,
            pseudo_random_generator,
        )

        private_mask = pseudo_random_generator(self.params.get("private_seed"))
        shared_mask = calculate_shared_mask(self.params.get("agreed_keys"))

        return private_mask, shared_mask

    def _save_mask_tensors(self):
        """
        Generates private and shared masks, stores them in a local tensor
        dictionary, and caches the dictionary in the tensor database.

        These tensors are then added to the gradient before sharing them
        with the aggregator during training task.
        """
        private_mask, shared_mask = self._generate_masks()
        local_tensor_dict = {
            TensorKey("private_mask", self.name, -1, False, ("secagg",)): [private_mask],
            TensorKey("shared_mask", self.name, -1, False, ("secagg",)): [shared_mask],
        }
        self.tensor_db.cache_tensor(local_tensor_dict)
        logger.info("SecAgg: setup completed, saved required tensors to db.")

    def _send_to_aggregator(self, tensor_dict: dict, stage: str):
        """
        Sends the provided tensor dictionary to the aggregator after
        compressing it.

        Args:
            tensor_dict (dict): A dictionary where keys are tensor names and
                values are numpy arrays.
            stage (str): The current stage of the secure aggregation process.
        """
        named_tensors = []
        # Convert python dict to tensor dict.
        for key, nparray in tensor_dict.items():
            tensor_key = TensorKey(
                key,
                self.name,
                -1,
                False,
                (
                    self.name,
                    "secagg",
                ),
            )
            named_tensor = utils.construct_named_tensor(
                tensor_key, str.encode(json.dumps(nparray)), {}, lossless=True
            )
            named_tensors.append(named_tensor)

        self.client.send_local_task_results(-1, f"secagg_{stage}", -1, named_tensors)

    def _fetch_from_aggregator(self, key_name):
        """
        Fetches the aggregated tensor data from a aggregator.

        Args:
            key_name (str): The name of the key to fetch the tensor for.

        Returns:
            bytes: The aggregated tensor data in bytes.
        """
        tensor = self.client.get_aggregated_tensor(key_name, -1, False, ("secagg",), True)
        return json.loads(tensor.data_bytes)
