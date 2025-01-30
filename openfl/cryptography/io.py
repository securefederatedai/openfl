# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Cryptography IO utilities."""

import os
from hashlib import sha384
from pathlib import Path
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509.base import Certificate, CertificateSigningRequest


def read_key(path: Path) -> RSAPrivateKey:
    """Reads a private key from a file.

    Args:
        path (Path): The path to the file containing the private key.

    Returns:
        RSAPrivateKey: The private key.
    """
    with open(path, "rb") as f:
        pem_data = f.read()

    signing_key = load_pem_private_key(pem_data, password=None)
    # TODO: replace assert with exception / sys.exit
    assert isinstance(signing_key, rsa.RSAPrivateKey)
    return signing_key


def write_key(key: RSAPrivateKey, path: Path) -> None:
    """Writes a private key to a file.

    Args:
        key (RSAPrivateKey): The private key to write.
        path (Path): The path to the file to write the private key to.
    """

    def key_opener(path, flags):
        return os.open(path, flags, mode=0o600)

    with open(path, "wb", opener=key_opener) as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )


def read_crt(path: Path) -> Certificate:
    """Reads a signed TLS certificate from a file.

    Args:
        path (Path): The path to the file containing the certificate.

    Returns:
        Certificate: The TLS certificate.
    """
    with open(path, "rb") as f:
        pem_data = f.read()

    certificate = x509.load_pem_x509_certificate(pem_data)
    # TODO: replace assert with exception / sys.exit
    assert isinstance(certificate, x509.Certificate)
    return certificate


def write_crt(certificate: Certificate, path: Path) -> None:
    """Writes a cryptography certificate / CSR to a file.

    Args:
        certificate (Certificate): cryptography csr / certificate object to
            write.
        path (Path): The path to the file to write the certificate to.
    """
    with open(path, "wb") as f:
        f.write(
            certificate.public_bytes(
                encoding=serialization.Encoding.PEM,
            )
        )


def read_csr(path: Path) -> Tuple[CertificateSigningRequest, str]:
    """Reads a Certificate Signing Request (CSR) from a file.

    Args:
        path (Path): The path to the file containing the CSR.

    Returns:
        Tuple[CertificateSigningRequest, str]: The CSR and its hash.
    """
    with open(path, "rb") as f:
        pem_data = f.read()

    csr = x509.load_pem_x509_csr(pem_data)
    # TODO: replace assert with exception / sys.exit
    assert isinstance(csr, x509.CertificateSigningRequest)
    return csr, get_csr_hash(csr)


def get_csr_hash(certificate: CertificateSigningRequest) -> str:
    """Computes the SHA-384 hash of a certificate.

    Args:
        certificate (CertificateSigningRequest): The certificate to compute
            the hash of.

    Returns:
        str: The SHA-384 hash of the certificate.
    """
    hasher = sha384()
    encoded_bytes = certificate.public_bytes(
        encoding=serialization.Encoding.PEM,
    )
    hasher.update(encoded_bytes)
    return hasher.hexdigest()
