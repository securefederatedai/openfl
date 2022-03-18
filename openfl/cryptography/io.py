# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cryptography IO utilities."""

from hashlib import sha384
from pathlib import Path
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509.base import Certificate
from cryptography.x509.base import CertificateSigningRequest


def read_key(path: Path) -> RSAPrivateKey:
    """
    Read private key.

    Args:
        path : Path (pathlib)

    Returns:
        private_key
    """
    with open(path, 'rb') as f:
        pem_data = f.read()

    signing_key = load_pem_private_key(pem_data, password=None)
    assert(isinstance(signing_key, rsa.RSAPrivateKey))
    return signing_key


def write_key(key: RSAPrivateKey, path: Path) -> None:
    """
    Write private key.

    Args:
        key  : RSA private key object
        path : Path (pathlib)

    """
    with open(path, 'wb') as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))


def read_crt(path: Path) -> Certificate:
    """
    Read signed TLS certificate.

    Args:
        path : Path (pathlib)

    Returns:
        Cryptography TLS Certificate object
    """
    with open(path, 'rb') as f:
        pem_data = f.read()

    certificate = x509.load_pem_x509_certificate(pem_data)
    assert(isinstance(certificate, x509.Certificate))
    return certificate


def write_crt(certificate: Certificate, path: Path) -> None:
    """
    Write cryptography certificate / csr.

    Args:
        certificate : cryptography csr / certificate object
        path : Path (pathlib)

    Returns:
        Cryptography TLS Certificate object
    """
    with open(path, 'wb') as f:
        f.write(certificate.public_bytes(
            encoding=serialization.Encoding.PEM,
        ))


def read_csr(path: Path) -> Tuple[CertificateSigningRequest, str]:
    """
    Read certificate signing request.

    Args:
        path : Path (pathlib)

    Returns:
        Cryptography CSR object
    """
    hasher = sha384()
    with open(path, 'rb') as f:
        pem_data = f.read()
        hasher.update(pem_data)

    csr = x509.load_pem_x509_csr(pem_data)
    assert(isinstance(csr, x509.CertificateSigningRequest))
    return csr, hasher.hexdigest()
