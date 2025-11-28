# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Cryptography participant utilities."""

from typing import Tuple

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509.base import CertificateSigningRequest
from cryptography.x509.oid import NameOID


def generate_csr(
    common_name: str, server: bool = False
) -> Tuple[RSAPrivateKey, CertificateSigningRequest]:
    """Issue a Certificate Signing Request (CSR) for a server or a client.

    Args:
        common_name (str): The common name for the certificate.
        server (bool, optional): A flag to indicate if the CSR is for a server.
            If False, the CSR is for a client. Defaults to False.

    Returns:
        Tuple[RSAPrivateKey, CertificateSigningRequest]: The private key and
            the CSR.
    """
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072, backend=default_backend()
    )

    builder = x509.CertificateSigningRequestBuilder()
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    builder = builder.subject_name(subject)
    builder = builder.add_extension(
        x509.BasicConstraints(ca=False, path_length=None),
        critical=True,
    )
    if server:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=True,
        )

    else:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=True,
        )

    builder = builder.add_extension(
        x509.KeyUsage(
            digital_signature=True,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            content_commitment=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=True,
    )

    builder = builder.add_extension(
        x509.SubjectAlternativeName([x509.DNSName(common_name)]), critical=False
    )

    # Sign the CSR
    csr = builder.sign(
        private_key=private_key,
        algorithm=hashes.SHA384(),
        backend=default_backend(),
    )

    return private_key, csr
