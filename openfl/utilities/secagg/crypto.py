# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
This file contains utility functions for Secure Aggregation's cipher related
operations.
"""

import random
from typing import Union

import numpy as np
from Crypto.Cipher import AES  # nosec B413
from Crypto.Util.Padding import pad  # nosec B413


def create_ciphertext(
    secret_key: bytes,
    source_id: int,
    destination_id: int,
    seed_share: str,
    key_share: str,
    nonce: bytes = b"nonce",
) -> tuple[bytes, bytes, bytes]:
    """
    Creates a cipher-text using a cipher_key for collaborators source_id and
    destination_id, and share of the private seed and share of the private key.

    The function creates a byte string using the args such that
    data = b'source_id.destination_id.seed_share.key_share'.
    The "." serves as a separator such that all values used to create the
    ciphertext can be easily distinguished when decrypting.

    Args:
        secret_key (bytes): Agreed key in bytes used to construct a cipher for
            the encryption.
        source_id (int): Unique integer ID of the creating collaborator of the
            cipher text.
        destination_id (int): Unique integer ID of the recipient collaborator
            of the cipher text.
        seed_share (bytes): Share of source_id collaborator's private seed for
            destination_id collaborator.
        key_share (bytes): Share of source_id collaborator's private key for
            destination_id collaborator.

    Returns:
        bytes: Ciphertext created using the args.
        bytes: MAC tag for the ciphertext which can be used for verification.
        bytes: Nonce used for generating the cipher used for decryption.
    """
    # Converting the integer collaborator IDs to bytes.
    source_id_bytes = source_id.to_bytes(4, byteorder="big")
    destination_id_bytes = destination_id.to_bytes(4, byteorder="big")
    # Generate the byte string to be encrypted.
    data = (
        source_id_bytes
        + b" "
        + destination_id_bytes
        + b" "
        + str.encode(seed_share)
        + b" "
        + str.encode(key_share)
    )
    # AES cipher requires the secret key to be of a certain length.
    # We use 64 bytes as it is the maximum length available.
    padded_secret_key = pad(secret_key, 64)

    from Crypto.Random import get_random_bytes  # nosec B413

    # Generate a random nonce to make the encryption non-deterministic.
    nonce = get_random_bytes(len(padded_secret_key) // 2)
    # Generate a ciphertext using symmetric block cipher.
    cipher = AES.new(padded_secret_key, AES.MODE_SIV, nonce=nonce)
    ciphertext, mac = cipher.encrypt_and_digest(data)

    return ciphertext, mac, nonce


def decipher_ciphertext(
    secret_key: bytes, ciphertext: bytes, mac: bytes, nonce: bytes
) -> tuple[int, int, bytes, bytes]:
    """
    Decrypt a cipher-text to get the values used to create it.

    The function uses the nonce used while creation of the ciphertext to
    create a cipher. This cipher is used to decypt the ciphertext and verify
    it using the MAC tag, which was also generated during creation of the
    ciphertext.

    Args:
        secret_key (bytes): Agreed key in bytes used to construct a cipher for
            the encryption.
        ciphertext (bytes): Ciphertext to be decrypted.
        mac (bytes): MAC tag for the ciphertext which is used for verification.
        nonce (bytes): Nonce used during cipher generation used for decryption.

    Returns:
        int: Unique integer ID of the creating collaborator of the ciphertext.
        int: Unique integer ID of the recipient collaborator of the ciphertext.
        bytes: Share of source_id collaborator's private seed for
            destination_id collaborator.
        bytes: Share of source_id collaborator's private key for
            destination_id collaborator.
    """
    # Recreate the secret key used for encryption by adding the extra padding.
    padded_secret_key = pad(secret_key, 64)
    # Generate a ciphertext using symmetric block cipher.
    cipher = AES.new(padded_secret_key, AES.MODE_SIV, nonce=nonce)

    data = cipher.decrypt_and_verify(ciphertext, mac)
    # Remove the separator " " from the decrypted data.
    # data = b'source_id destination_id seed_share key_share'
    data = data.split(b" ")

    return (
        # Convert the collaborator IDs to int.
        int.from_bytes(data[0], "big"),
        int.from_bytes(data[1], "big"),
        data[2],
        data[3],
    )


def pseudo_random_generator(seed: Union[int, float, bytes]) -> np.ndarray:
    """
    Generates a random mask using a seed value passed as arg.

    Args:
        seed (Union[int, float, bytes]): Seed to be used for generating a
            pseudo-random number.
        shape (Tuple): Shape of the numpy array to be generated.

    Returns:
        np.ndarray: array with pseudo-randomly generated numbers.
    """
    # Seed random generator.
    random.seed(seed)

    return random.random()


def calculate_shared_mask(agreed_keys: list) -> float:
    """
    Calculate the shared mask based on a list of agreed keys.

    Args:
        agreed_keys (list): A list of tuples where each tuple contains three
            elements:
            - source_index (int): The index of the source.
            - dest_index (int): The index of the destination.
            - agreed_key (Any): The agreed key used for generating the mask.

    Returns:
        float: The total shared mask calculated by adding or subtracting the
            pseudo-random values generated from the agreed keys based on the
            comparison of source and destination indices.
    """
    total_mask = 0.0
    for key in agreed_keys:
        source_index = key[0]
        dest_index = key[1]
        agreed_key = key[2]
        if source_index > dest_index:
            total_mask += pseudo_random_generator(agreed_key)
        elif source_index == dest_index:
            continue
        else:
            total_mask -= pseudo_random_generator(agreed_key)

    return total_mask


def calculate_mask(collaborator_index, agreed_keys, private_seed) -> float:
    """
    Calculates the sum of shared masks between collaborators using
    the agreed keys as the seed for each mask and adding private mask to it.
    """
    total_mask = 0.0
    # Calculating the sum of all shared masks.
    for index in agreed_keys:
        # Using second agreed key to use as seed.
        key = agreed_keys[index][1]
        random.seed(key)
        seed = random.random()
        # mask_u,v = ∆_u,v * PRG(agreed2_u,v) such that
        # ∆_u,v = 1 if u > v else ∆_u,v = -1 if u < v
        # and ∆_u,v = 0 if u = v
        if collaborator_index > index:
            total_mask += seed
        elif collaborator_index < index:
            total_mask -= seed
    # Calculating the private mask for the collaborator.
    random.seed(private_seed)
    total_mask += random.random()

    return total_mask
