# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Cryptography CA utilities."""

import datetime
import uuid
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509.base import Certificate, CertificateSigningRequest
from cryptography.x509.extensions import ExtensionNotFound
from cryptography.x509.name import Name
from cryptography.x509.oid import ExtensionOID, NameOID


def generate_root_cert(
    days_to_expiration: int = 365,
) -> Tuple[RSAPrivateKey, Certificate]:
    """Generate a root certificate and its corresponding private key.

    Args:
        days_to_expiration (int, optional): The number of days until the
            certificate expires. Defaults to 365.

    Returns:
        Tuple[RSAPrivateKey, Certificate]: The private key and the certificate.
    """
    now = datetime.datetime.utcnow()
    expiration_delta = days_to_expiration * datetime.timedelta(1, 0, 0)

    # Generate private key
    root_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072, backend=default_backend()
    )

    # Generate public key
    root_public_key = root_private_key.public_key()
    builder = x509.CertificateBuilder()
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.DOMAIN_COMPONENT, "org"),
            x509.NameAttribute(NameOID.DOMAIN_COMPONENT, "simple"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Simple Root CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Simple Inc"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Simple Root CA"),
        ]
    )
    issuer = subject
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)

    builder = builder.not_valid_before(now)
    builder = builder.not_valid_after(now + expiration_delta)
    builder = builder.serial_number(int(uuid.uuid4()))
    builder = builder.public_key(root_public_key)
    builder = builder.add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    )

    # Sign the CSR
    certificate = builder.sign(
        private_key=root_private_key,
        algorithm=hashes.SHA384(),
        backend=default_backend(),
    )

    return root_private_key, certificate


def generate_signing_csr() -> Tuple[RSAPrivateKey, CertificateSigningRequest]:
    """Generate a Certificate Signing Request (CSR) and its corresponding
    private key.

    Returns:
        Tuple[RSAPrivateKey, CertificateSigningRequest]: The private key and
            the CSR.
    """
    # Generate private key
    signing_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072, backend=default_backend()
    )

    builder = x509.CertificateSigningRequestBuilder()
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.DOMAIN_COMPONENT, "org"),
            x509.NameAttribute(NameOID.DOMAIN_COMPONENT, "simple"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Simple Signing CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Simple Inc"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Simple Signing CA"),
        ]
    )
    builder = builder.subject_name(subject)
    builder = builder.add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    )

    # Sign the CSR
    csr = builder.sign(
        private_key=signing_private_key,
        algorithm=hashes.SHA384(),
        backend=default_backend(),
    )

    return signing_private_key, csr


def sign_certificate(
    csr: CertificateSigningRequest,
    issuer_private_key: RSAPrivateKey,
    issuer_name: Name,
    days_to_expiration: int = 365,
    ca: bool = False,
) -> Certificate:
    """
    Sign a incoming Certificate Signing Request (CSR) with the issuer's
    private key.

    Args:
        csr (CertificateSigningRequest): The CSR to be signed.
        issuer_private_key (RSAPrivateKey): Root CA private key if the request
            is for the signing CA; Signing CA private key otherwise.
        issuer_name (Name): The name of the issuer.
        days_to_expiration (int, optional): The number of days until the
            certificate expires. Defaults to 365.
        ca (bool, optional): Whether the certificate is for a certificate
            authority (CA). Defaults to False.

    Returns:
        Certificate: The signed certificate.
    """
    now = datetime.datetime.utcnow()
    expiration_delta = days_to_expiration * datetime.timedelta(1, 0, 0)

    builder = x509.CertificateBuilder()
    builder = builder.subject_name(csr.subject)
    builder = builder.issuer_name(issuer_name)
    builder = builder.not_valid_before(now)
    builder = builder.not_valid_after(now + expiration_delta)
    builder = builder.serial_number(int(uuid.uuid4()))
    builder = builder.public_key(csr.public_key())
    builder = builder.add_extension(
        x509.BasicConstraints(ca=ca, path_length=None),
        critical=True,
    )
    try:
        builder = builder.add_extension(
            csr.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME).value,
            critical=False,
        )
    except ExtensionNotFound:
        pass  # Might not have alternative name

    signed_cert = builder.sign(
        private_key=issuer_private_key,
        algorithm=hashes.SHA384(),
        backend=default_backend(),
    )
    return signed_cert
