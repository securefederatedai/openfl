# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
This file contains utility functions for Secure Aggregation Shamir's secret
sharing related operations.
"""

from Crypto.Protocol.SecretSharing import Shamir  # nosec B413

# The length of each chunk of secret used for creation of shares.
SECRET_CHUNK_LENGTH = 16


def create_secret_shares(
    secret: bytes,
    count: int,
    threshold: int,
) -> dict:
    """
    Using Shamir's secret sharing protocol, creates shares for the secret.
    pycryptodome's Shamir.split requires the secret to be of length
    SECRET_CHUNK_LENGTH. Since the secret length cannot always be set to be
    SECRET_CHUNK_LENGTH, we split the secret byte string into multiple byte
    string each of length SECRET_CHUNK_LENGTH and create shares for each of
    the SECRET_CHUNK_LENGTH chunks.
    During reconstruction, the SECRET_CHUNK_LENGTH chunks are recreated using
    the respective shares and then concatenated to form the original secret.
    Thus each of the collaborator indices have multiple shares (one for each
    chunk) for a single secret.

    Args:
        secret (bytes): Secret for which shares are to be created.
        count (int): Number of shares to be created for the secret.
        threshold (int): Minimum threshold of shares required for
            reconstruction of the secret.

    Returns:
        dict: Contains the mapping of the index to the shares that belong to
            the respective index.
    """
    shares = {}

    from Crypto.Util.Padding import pad  # nosec B413

    # Pad the secret to create a byte string of a length which is a multiple
    # of SECRET_CHUNK_LENGTH.
    secret = pad(secret, SECRET_CHUNK_LENGTH)
    # Divide the secret into multiple chunks.
    secret_chunks = [
        secret[i : i + SECRET_CHUNK_LENGTH] for i in range(0, len(secret), SECRET_CHUNK_LENGTH)
    ]
    # Create shares for each of the chunk.
    for chunk in secret_chunks:
        chunk_shares = Shamir.split(threshold, count, chunk)
        # Map the respective chunk share to the id it belongs to.
        for id, share in chunk_shares:
            if id not in shares:
                shares[id] = []
            shares[id].append(share.hex())

    shares = {k: ".".join(v) for k, v in shares.items()}

    return shares


def reconstruct_secret(shares: dict) -> bytes:
    """
    Args:
        shares (dict): Contains the mapping of the index to the chunk shares
            that belong to the respective index.

    Returns:
        bytes: Secret reconstructed from the chunk shares.
    """
    secret = b""

    shares = {k: v.split(".") for k, v in shares.items()}

    total_chunks = max(len(share) for share in shares.values())
    for chunk_index in range(total_chunks):
        # Create a list for the respective chunk with all the shares.
        chunk_shares = [(key, bytes.fromhex(shares[key][chunk_index])) for key in shares]
        # Reconstruct the chunk of the secret.
        secret_chunk = Shamir.combine(chunk_shares)
        # Concatenate the chunk to the secret.
        secret += secret_chunk

    from Crypto.Util.Padding import unpad  # nosec B413

    # Remove the padding from the secret.
    secret = unpad(secret, SECRET_CHUNK_LENGTH)

    return secret
