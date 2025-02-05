# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
This file contains callbacks that help setup for secure aggregation for
both, the aggregator and collaborator.
"""
import logging
import struct
import time

import numpy as np

from openfl.callbacks.callback import Callback
from openfl.utilities import TensorKey
from openfl.utilities.secagg import (
    calculate_shared_mask,
    create_ciphertext,
    create_secret_shares,
    decipher_ciphertext,
    generate_agreed_key,
    generate_key_pair,
    pseudo_random_generator,
    reconstruct_secret,
)


logger = logging.getLogger(__name__)


class CollaboratorSecAgg(Callback):
    """
    This callback is used by the collaborator to perform the setup steps
    for scure aggregation on the collaborators.

    Required params include:
    - col_name: Name of the collaborator using the callback.
    - aggregator_secagg: Client for aggregator secure aggregation setup.

    It also requires the tensor-db client to be set.
    """
    def on_experiment_begin(self):
        """
        Used to perform secure aggregation setup before experiment begins.
        """
        self.name = self.params["col_name"]
        self.aggregator_secagg = self.params["aggregator_secagg"]
        logger.info(
            "Secure aggregation is enabled, starting setup..."
        )
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
        self._save_tensors()

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
        private_key1, public_key1 = generate_key_pair()
        private_key2, public_key2 = generate_key_pair()

        local_result = {
            "private_key": [private_key1, private_key2],
            "public_key": [public_key1, public_key2],
            "private_seed": np.random.random()
        }
        global_results = {
            "public_key": [public_key1, public_key2],
        }

        self.aggregator_secagg.send_to_participant(self.name, global_results)
        # Update callback params as the results for this step are reused at a
        # later stage.
        self.params.update(local_result)
        logger.debug("SecAgg: Generate key-pair generation successful")

    def _fetch_public_keys(self):
        """
        Fetches public keys from participants and identifies the index of the
        current participant's public key.

        This method retrieves the public keys from the aggregator's secure
        aggregation mechanism. It then iterates through the fetched public
        keys to find the index of the current participant's public key based
        on the provided parameters.

        Returns:
            dict: A dictionary containing the public keys of all participants,
                where the keys are the participant indices and the values are
                the public keys.
        """
        public_keys = self.aggregator_secagg.fetch_from_participant("public_keys")
        for index in public_keys:
            if public_keys[index][0] == self.params["public_key"][0]:
                self.index = int(index)
                break

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
        logger.debug(
            "SecAgg: Generating ciphertexts to be shared with other collaborators"
        )
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

        global_results = {"ciphertext": [], "agreed_keys": []}
        local_result = {"ciphertext_verification": {}}
        # Create cipher-texts for each collaborator.
        for collab_index in public_keys:
            agreed_key = generate_agreed_key(
                private_keys[0],
                public_keys[collab_index][0]
            )
            ciphertext, mac, nonce = create_ciphertext(
                agreed_key,                 # agreed key
                self.index,                 # source collaborator index
                collab_index,               # destination collaborator index
                seed_shares[collab_index],  # seed share from source to dest
                key_shares[collab_index]    # key share from source to dest
            )
            global_results["ciphertext"].append(
                [self.index, collab_index, ciphertext]
            )
            local_result["ciphertext_verification"][collab_index] = [
                ciphertext, mac, nonce
            ]
            local_result["agreed_keys"].append(
                [self.index, collab_index, agreed_key]
            )

        self.aggregator_secagg.send_to_participant(self.name, global_results)
        # Update callback params as the results for this step are reused at a
        # later stage.
        self.params.update(local_result)

        logger.debug(
            "SecAgg: Ciphertexts shared with the aggregator successfully"
        )

    def _decrypt_ciphertexts(self, public_keys):
        """
        Decrypts the ciphertexts received from participants using the provided
        public keys.

        This method fetches the ciphertexts from the aggregator, decrypts them
        using the participant's private key and the provided public keys, and
        then sends the decrypted seed shares and key shares back to the
        participants.

        Args:
            public_keys (dict): A dictionary containing the public keys of the
                participants.
        """
        logger.debug(
            "SecAgg: fetching addressed ciphertexts from the aggregator"
        )

        ciphertexts = self.aggregator_secagg.fetch_from_participant("ciphertexts")
        private_keys = self.params["private_key"]
        ciphertext_verification = self.params["ciphertext_verification"]

        global_results = {
            "seed_share": {},
            "key_share": {}
        }

        for cipher in ciphertexts:
            source_index = cipher[0]
            if cipher[1] == self.index:
                _, _, seed_share, key_share = decipher_ciphertext(
                    generate_agreed_key(
                        private_keys[0],
                        public_keys[source_index][0]

                    ),
                    ciphertext_verification[source_index][0],
                    ciphertext_verification[source_index][1],
                    ciphertext_verification[source_index][2],
                )
                global_results["seed_share"][source_index] = [self.index, seed_share]
                global_results["key_share"][source_index] = [self.index, key_share]

        self.aggregator_secagg.send_to_participant(self.name, global_results)

        logger.debug(
            "SecAgg: decrypted ciphertexts shared with the aggregator"
        )

    def _generate_masks(self):
        """
        Use the private seed and agreed keys to calculate the masks to be
        added to the gradients.
        """
        private_mask = pseudo_random_generator(self.params.get("private_seed"))
        shared_mask = calculate_shared_mask(self.params.get("agreed_keys"))

        return private_mask, shared_mask

    def _save_tensors(self):
        """
        Generates private and shared masks, stores them in a local tensor
        dictionary, and caches the dictionary in the tensor database.

        These tensors are then added to the gradient before sharing them
        with the aggregator during trainign task.

        This method performs the following steps:
        1. Generates private and shared masks by calling the `_generate_masks`
            method.
        2. Creates a local tensor dictionary with the generated masks.
        3. Caches the local tensor dictionary in the tensor database.
        4. Logs an informational message indicating the completion of the
            setup and the saving of required tensors to the database.
        """
        private_mask, shared_mask = self._generate_masks()
        local_tensor_dict = {
            TensorKey(
                "private_mask", self.name, -1, False, ("secagg", )
            ): [private_mask],
            TensorKey(
                "shared_mask", self.name, -1, False, ("secagg", )
            ): [shared_mask],
        }
        self.tensor_db.cache_tensor(local_tensor_dict)
        logger.info(
            "SecAgg: setup completed, saved required tensors to db."
        )


class AggregatorSecAgg(Callback):
    """
    This callback is used by the aggregator to perform the setup steps
    for secure aggregation on the aggregator.

    Required params include:
    - col_name: Name of the collaborator using the callback.
    - aggregator_secagg: Client for aggregator secure aggregation setup.

    It also requires the tensor-db client to be set.
    """
    def on_experiment_begin(self):
        """
        Used to perform secure aggregation setup before experiment begins.

        Initializes results, waits for all collaborators to send their public
        keys, sorts public keys, ciphertexts, and secret shares, reconstructs
        secrets, generates agreed keys, and saves tensors.
        """
        logger.info(
            "Secure aggregation is enabled, starting setup..."
        )
        # Initialize results dictionary and collaborator list.
        self._results = {}
        self.collaborator_list = self.params["collaborators"]

        # Wait for all collaborators to send their public key.
        self._wait_for_all_collaborators("public_key", timeout=120)
        # Sort the received public keys.
        self._sort_public_keys()

        # Wait for all collaborators to send their ciphertexts.
        self._wait_for_all_collaborators("ciphertext", timeout=120)
        # Sort the received ciphertexts.
        self._sort_ciphertexts()

        # Wait for all collaborators to send their seed shares.
        self._wait_for_all_collaborators("seed_share", timeout=120)
        # Wait for all collaborators to send their key shares.
        self._wait_for_all_collaborators("key_share", timeout=120)
        # Sort the received secret shares (seed shares and key shares).
        self._sort_secret_shares()

        # Reconstruct the private seeds and private keys from the secret
        # shares.
        self._reconstruct_secrets()
        # Generate agreed keys between all pairs of collaborators.
        self._generate_agreed_keys()
        # Save the local tensors to the tensor database.
        self._save_tensors()

    def _wait_for_all_collaborators(self, key_name, timeout=120):
        """
        Waits for all collaborators to send their data for a given key.

        Args:
            key_name (str): The name of the key to wait for.
            timeout (int): The maximum time to wait for the data (in seconds).
        """
        start_time = time.time()
        while True:
            time.sleep(5)
            if key_name not in self._results:
                continue
            all_received = True
            for collaborator in self.collaborator_list:
                if collaborator not in self._results[key_name]:
                    all_received = False
            # Break out of loop if all collaborators have sent data.
            if all_received:
                break
            # Timeout
            if (time.time() - start_time) > timeout:
                break
        logger.debug(
            "SecAgg: received %s from all collaborators", key_name
        )

    def _sort_public_keys(self):
        """
        Sorts the public keys received from collaborators and updates the
        results.
        """
        self._results["public_keys"] = {}
        self._results["index"] = {}
        index = 1
        for col_name in self._results.get("public_key", {}):
            self._results["public_keys"][index] = self._results.get(
                "public_key", {}
            )[col_name]
            self._results["index"][col_name] = index
            index += 1

        self._results.pop("public_key", None)

    def _sort_ciphertexts(self):
        """
        Sorts the ciphertexts received from collaborators and updates the
        results.
        """
        self._results["ciphertexts"] = []
        for _, ciphertext in self._results.get("ciphertext", {}).items():
            for c in ciphertext:
                self._results["ciphertexts"].append(c)

        self._results.pop("ciphertext", None)

    def _sort_secret_shares(self):
        """
        Sorts the secret shares (seed shares and key shares) received from
        collaborators and updates the results.
        """
        self._results["seed_shares"] = {}
        for _, share_dict in self._results.get("seed_share", {}).items():
            for share_index, share in share_dict.items():
                if share[0] not in self._results["seed_shares"]:
                    self._results["seed_shares"][share[0]] = {}
                self._results["seed_shares"][share[0]][share_index] = share[1]

        self._results.pop("seed_share", None)

        self._results["key_shares"] = {}
        for _, share_dict in self._results.get("key_share", {}).items():
            for share_index, share in share_dict.items():
                if share[0] not in self._results["key_shares"]:
                    self._results["key_shares"][share[0]] = {}
                self._results["key_shares"][share[0]][share_index] = share[1]

        self._results.pop("key_share", None)

    def _reconstruct_secrets(self):
        """
        Reconstructs the private seeds and private keys from the secret shares.
        """
        self._results["private_seeds"] = {}
        self._results["private_keys"] = {}

        for source_id in self._results["seed_shares"]:
            self._results["private_seeds"][source_id] = reconstruct_secret(
                self._results["seed_shares"][source_id]
            )
            self._results["private_keys"][source_id] = reconstruct_secret(
                self._results["key_shares"][source_id]
            )
        logger.info(
            "SecAgg: recreated secrets successfully"
        )

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
                self._results["agreed_keys"].append([
                    source_index,
                    dest_index,
                    generate_agreed_key(
                        self._results["private_keys"][source_index],
                        self._results["public_keys"][dest_index][0],
                    )
                ])

    def _generate_masks(self):
        """
        Use the private seeds and agreed keys to calculate the masks to be
        removed from gradient aggregate.
        """
        private_mask_sum = 0.0
        for seed in self._results["private_seeds"].values():
            private_mask_sum += pseudo_random_generator(seed)

        shared_mask_sum = calculate_shared_mask(self._results["agreed_keys"])

        return private_mask_sum, shared_mask_sum

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
        private_mask_sum, shared_mask_sum = self._generate_masks()
        local_tensor_dict = {
            # TensorKey(
            #     "private_seeds", "agg", -1, False, ("secagg", )
            # ): [
            #     [index, seed]
            #     for index, seed in self._results["private_seeds"].items()
            # ],
            # TensorKey(
            #     "agreed_keys", "agg", -1, False, ("secagg", )
            # ): self._results["agreed_keys"],
            TensorKey(
                "masks_sum", "agg", -1, False, ("secagg", )
            ): [private_mask_sum, shared_mask_sum],

        }
        self.tensor_db.cache_tensor(local_tensor_dict)
        logger.info(
            "SecAgg: setup completed, saved required tensors to db."
        )

    # TODO: Replace with actual server/client usage.
    def send_to_participant(self, col_name, data):
        """
        Sends data to a participant.

        Args:
            col_name (str): The name of the collaborator to send data to.
            data (dict): The data to send to the collaborator.
        """
        for key, value in data.items():
            if key not in self._results:
                self._results[key] = {}
            self._results[key][col_name] = value

    # TODO: Replace with actual server/client usage.
    def fetch_from_participant(self, key_name):
        """
        Fetches data from a participant.

        Args:
            key_name (str): The name of the key to fetch data for.

        Returns:
            dict: The data fetched from the participant.
        """
        start_time = time.time()
        while True:
            time.sleep(5)
            if key_name in self._results:
                break
            # Timeout
            if (time.time() - start_time) > 120:
                break

        return self._results.get(key_name, {})
