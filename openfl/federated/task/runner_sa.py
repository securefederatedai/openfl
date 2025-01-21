# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import random
import struct

import numpy as np

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import TensorKey
from openfl.utilities.secagg import (
    create_ciphertext,
    create_secret_shares,
    decipher_ciphertext,
    generate_agreed_key,
    generate_key_pair,
    pseudo_random_generator,
)


class SATaskRunner(TaskRunner):
    """
    NOTE: This class is only for testing purposes.
    The tasks/methods define din this class would be placed in TaskRunner
    class. By doing so, the users would be able to leverage secure aggregation
    using a framework of their choice: torch, keras, etc.
    """
    def __init__(self, device: str = None, loss_fn=None, optimizer=None, **kwargs):
        super().__init__(self, **kwargs)
        self.required_tensorkeys_for_function = {}

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """
        Initialize the required TensorKeys for various functions.

        This method sets up the required TensorKeys for the functions 
        "generate_keys", "generate_ciphertexts", and "decrypt_ciphertexts". 
        The TensorKeys specify the name, scope, version, and other attributes 
        for the tensors used in these functions.

        Args:
            with_opt_vars (bool): If True, includes optional variables in the 
                                  TensorKeys. Defaults to False.

        TensorKeys:
            - "generate_keys": No TensorKeys required.
            - "generate_ciphertexts":
                - public_key: A global tensor key for the public key.
                - public_key_local: A local tensor key for the public key.
            - "decrypt_ciphertexts":
                - ciphertext: A global tensor key for the ciphertext.
                - ciphertext_local: A local tensor key for the ciphertext.
                - index: A local tensor key for the index.
        """
        self.required_tensorkeys_for_function["generate_keys"] = []
        self.required_tensorkeys_for_function["generate_ciphertexts"] = [
            TensorKey("public_key", "GLOBAL", 1, False, ("public_key")),
            TensorKey("public_key_local", "LOCAL", 1, False, ("public_key")),
        ]
        self.required_tensorkeys_for_function["decrypt_ciphertexts"] = [
            TensorKey("ciphertext", "GLOBAL", 1, False, ("ciphertext")),
            TensorKey("ciphertext_local", "LOCAL", 1, False, ("ciphertext_local")),
            TensorKey("index", "LOCAL", 1, False, ())
        ]

    def generate_keys(
        self,
        col_name,
        round_number,
        input_tensor_dict: dict,
        **kwargs,
    ):
        """
        Generates a pair of private and public keys and returns them in
        dictionaries.

        Args:
            col_name (str): The column name associated with the keys.
            round_number (int): The round number for which the keys are
                generated.
            input_tensor_dict (dict): A dictionary of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing two dictionaries:
                - local_tensor_dict (dict): A dictionary with private keys and
                    a random seed.
                - global_tensor_dict (dict): A dictionary with public keys.
        """
        private_key1, public_key1 = generate_key_pair()
        private_key2, public_key2 = generate_key_pair()

        local_tensor_dict = {
            TensorKey("private_key", col_name, round_number, False, ("private_key")): [
                private_key1,
                private_key2,
            ],
            TensorKey("public_key_local", col_name, round_number, False, ("public_key")): [
                public_key1,
                public_key2,
            ],
            TensorKey("private_seed", col_name, round_number, False, ()): [random.random()],
        }

        global_tensor_dict = {
            TensorKey("public_key", col_name, round_number, False, ("public_key")): [
                public_key1,
                public_key2,
            ]
        }

        return local_tensor_dict, global_tensor_dict

    def generate_ciphertexts(
        self,
        col_name,
        round_number,
        input_tensor_dict: dict,
        **kwargs,
    ):
        """
        Generates ciphertexts for secure multi-party computation.

        This method generates ciphertexts for each collaborator using their
        public keys and the local private key. It creates secret shares for a
        private seed and the private key, then generates agreed keys and
        ciphertexts for each collaborator.

        Required tensors for the task include:
        - GLOBAL public_key
        - LOCAL public_key_local

        Args:
            col_name (str): The column name for the tensor key.
            round_number (int): The current round number.
            input_tensor_dict (dict): A dictionary containing the required
                tensors:
                - "public_key": List of public keys for all collaborators.
                - "public_key_local": The local public key.

        Returns:
            tuple: A tuple containing two dictionaries:
                - The first dictionary contains the global output with the
                    tensor key for ciphertexts.
                - The second dictionary contains the local output with tensor
                    keys for local ciphertexts and the index of the current
                    collaborator.
        """
        global_output = []
        local_output = []
        # public_key is in the format.
        # [
        #     [collaborator_index, public_key_1, public_key_2],
        #     [collaborator_index, public_key_1, public_key_2],
        #     ...
        # ]
        public_keys = input_tensor_dict["public_key"]
        # Get the total number of collaborators participating.
        collaborator_count = len(public_keys)
        # Get the index of the collaborator by matching the public key.
        index_current = -1
        for tensor in public_keys:
            if tensor[1] == input_tensor_dict["public_key_local"][0]:
                index_current = tensor[0]
                break
        # Generate a private seed for the collaborator.
        private_seed = random.random()
        # Create secret shares for the private seed.
        seed_shares = create_secret_shares(
            # Converts the floating-point number private_seed into an 8-byte
            # binary representation.
            struct.pack("d", private_seed),
            collaborator_count,
            collaborator_count,
        )
        # Create secret shares for the private key.
        key_shares = create_secret_shares(
            str.encode(self._private_keys[0].export_key(format="PEM")),
            collaborator_count,
            collaborator_count,
        )
        # Create cipher-texts for each collaborator.
        for collaborator_tensor in public_keys:
            collab_index = collaborator_tensor[0]
            collab_public_key_1 = collaborator_tensor[1]
            collab_public_key_2 = collaborator_tensor[2]
            # Generate agreed keys for both the public keys.
            agreed_key_1 = generate_agreed_key(self._private_keys[0], collab_public_key_1)
            agreed_key_2 = generate_agreed_key(self._private_keys[1], collab_public_key_2)
            # Generate ciphertext for the collaborator.
            ciphertext, mac, nonce = create_ciphertext(
                agreed_key_1,
                index_current,
                collab_index,
                seed_shares[collab_index],
                key_shares[collab_index],
            )
            # Local cache for collaborator ID x contains a list which contains
            # [x, ciphertext_for_x, mac_for_x, nonce_for_x,
            #   agreed_key_1_with_x, agreed_key_2_with_x].
            local_output.append([collab_index, ciphertext, mac, nonce, agreed_key_1, agreed_key_2])
            # Result sent to aggregator contains a row for each collaborator
            # such that [source_id, destination_id, ciphertext_source_to_dest].
            global_output.append([index_current, collab_index, ciphertext])

        return {
            TensorKey("ciphertext", col_name, round_number, False, ("ciphertext")): global_output
        }, {
            TensorKey(
                "ciphertext_local", col_name, round_number, False, ("ciphertext")
            ): local_output,
            TensorKey("index", col_name, round_number, False, ()): [index_current],
        }

    def decrypt_ciphertexts(
        self,
        col_name,
        round_number,
        input_tensor_dict: dict,
        **kwargs,
    ):
        """
        Decrypts the provided ciphertexts and returns the deciphered outputs.

        Required tensors for the task include:
        - GLOBAL ciphertext.
        - LOCAL ciphertext_local
        - LOCAL index

        Args:
            col_name (str): The name of the column.
            round_number (int): The current round number.
            input_tensor_dict (dict): A dictionary containing the required
                tensors:
                - "ciphertext": List of ciphertexts in the format
                  [[source_collaborator_id, destination_collaborator_id,
                    ciphertext], ...].
                - "index": The current index.
                - "ciphertext_local": Local ciphertext information.

        Returns:
            tuple: A tuple containing:
                - dict: A dictionary with the key as TensorKey and value as
                    the list of deciphered outputs.
                - dict: An empty dictionary (reserved for future use).
        """
        global_output = []
        # ciphertexts is in format
        # [
        #     [source_collaborator_id, destination_collaborator_id, ciphertext]
        #     [source_collaborator_id, destination_collaborator_id, ciphertext]
        #     ...
        # ]
        ciphertexts = input_tensor_dict["ciphertext"]
        index_current = input_tensor_dict["index"]
        ciphertext_local = input_tensor_dict["ciphertext_local"]
        addressed_ciphertexts = self._filter_ciphertexts(ciphertexts, index_current)

        for ciphertext in addressed_ciphertexts:
            source_index = ciphertext[0]
            cipher_details = self._fetch_collaborator_ciphertext(source_index, ciphertext_local)
            _, _, seed_share, key_share = decipher_ciphertext(
                cipher_details[4],  # agreed_key_1
                ciphertext[2],  # ciphertext
                cipher_details[2],  # mac
                cipher_details[3],  # nonce
            )
            # Result sent to aggregator contains a row for each collaborator
            # such that [source_id, destination_id, source_seed_dest_share,
            # source_key_dest_share].
            global_output.append(source_index, index_current, seed_share, key_share)

        return {
            TensorKey(
                "deciphertext", col_name, round_number, False, ("deciphertext")
            ): global_output
        }, {}

    def _filter_ciphertexts(self, ciphertexts, index_current):
        """
        Filters the given list of ciphertexts to include only those that match
        the specified index.

        Args:
            ciphertexts (list): A list of ciphertexts, where each ciphertext
                is expected to be a list or tuple.
            index_current (int): The index to filter the ciphertexts by.

        Returns:
            list: A list of filtered ciphertexts that match the specified
                index.
        """
        # ciphertexts contains a list of global ciphertext values such that
        # each row contains
        # [source_id, destination_id, ciphertext_source_to_dest]
        filtered_ciphertexts = []
        for ciphertext in ciphertexts:
            # Filter the ciphertexts addressed to the "index_current"
            # collaborator.
            if ciphertext[1] == index_current:
                filtered_ciphertexts.append(ciphertext)

        return filtered_ciphertexts

    def _fetch_collaborator_ciphertext(self, collaborator_id, ciphertext_local):
        """
        Fetches the ciphertext associated with a specific collaborator.

        Args:
            collaborator_id (str): The ID of the collaborator whose ciphertext
                is to be fetched.
            ciphertext_local (list): A list of ciphertexts, where each
                ciphertext is a tuple containing a collaborator ID and the
                corresponding ciphertext.

        Returns:
            The ciphertext associated with the given collaborator ID, or None
                if no match is found.
        """
        # ciphertext_local contains a list of ciphertext related values in a
        # list format. This list is formatted such that for dest collaborator x
        # [x, ciphertext_for_x, mac_for_x, nonce_for_x,
        #   agreed_key_1_with_x, agreed_key_2_with_x].
        for ciphertext in ciphertext_local:
            # Filter the ciphertexts for collaborator_id.
            if ciphertext[0] == collaborator_id:
                return ciphertext

    def _mask(
        self,
        index,
        private_seed,
        ciphertext_local,
        gradient: np.ndarray,
    ):
        """
        Apply a mask to the gradient using shared and private seeds.

        This function modifies the input gradient by adding masks generated
        from shared keys and a private seed. The shared masks are generated
        using the keys from `ciphertext_local` and the private mask is
        generated using the `private_seed`.

        Args:
            index (int): The index of the current collaborator.
            private_seed (Any): The private seed used to generate the private
                mask.
            ciphertext_local (list): A list of ciphertexts, where each
                ciphertext contains information about the shared keys and
                indices of collaborators.
            gradient (np.ndarray): The gradient to be masked.

        Returns:
            np.ndarray: The masked gradient.
        """
        shape = gradient.shape()

        for ciphertext in ciphertext_local:
            # ciphertext[4] is the agreed key for the two collaborators.
            shared_mask = pseudo_random_generator(ciphertext[4], shape)
            if index < ciphertext[0]:
                shared_mask = np.dot(-1, shared_mask)
            elif index == ciphertext[0]:
                shared_mask = np.dot(0, shared_mask)
            # Add masks for all the collaborators.
            gradient = np.add(gradient, shared_mask)

        # Generate private mask for the collaborator.
        private_mask = pseudo_random_generator(private_seed, shape)
        gradient = np.add(gradient, private_mask)

        return gradient
