# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
This file contains utility functions for Secure Aggregation for key operations.
"""

from Crypto.PublicKey import ECC  # nosec B413


def generate_key_pair(curve: str = "ed25519") -> tuple[str, str]:
    """
    Generates a public-private key pair for a specific curve.

    Args:
        curve (str, optional): The curve to use for generating the key pair.
            Defaults to 'ed25519'

    Returns:
        str: Private key in string.
        str: Public key in string.
    """
    # Generate private key.
    private_key = ECC.generate(curve=curve)
    # Generate public_key
    public_key = private_key.public_key().export_key(format="PEM")

    return private_key.export_key(format="PEM"), public_key


def generate_agreed_key(
    private_key: bytes,
    public_key: bytes,
) -> bytes:
    """
    Uses Diffie-Helman key agreement to generate an agreed key between a pair
    of public-private keys.

    Args:
        public_key (bytes): Public key to be used for key agreement.
        private_key (bytes): Private key to be used for key agreement.

    Returns:
        bytes: Agreed key between the two keys shared in args.
    """
    from Crypto.Hash import SHAKE128  # nosec B413

    # Key derivation function.
    def kdf(x):
        return SHAKE128.new(x).read(32)

    from Crypto.Protocol.DH import key_agreement  # nosec B413

    # Using Diffie-Hellman key agreement.
    key = key_agreement(
        static_priv=ECC.import_key(private_key), static_pub=ECC.import_key(public_key), kdf=kdf
    )

    return key
