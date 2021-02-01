# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cryptography CA utilities."""

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import datetime
import uuid


def generate_root_cert(days_to_expiration=365):
    """Generate_root_certificate."""
    now = datetime.datetime.utcnow()
    expiration_delta = days_to_expiration * datetime.timedelta(1, 0, 0)

    # Generate private key
    root_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=3072,
        backend=default_backend()
    )

    # Generate public key
    root_public_key = root_private_key.public_key()
    builder = x509.CertificateBuilder()
    subject = x509.Name([
        x509.NameAttribute(NameOID.DOMAIN_COMPONENT, u'org'),
        x509.NameAttribute(NameOID.DOMAIN_COMPONENT, u'simple'),
        x509.NameAttribute(NameOID.COMMON_NAME, u'Simple Root CA'),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u'Simple Inc'),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, u'Simple Root CA'),
    ])
    issuer = subject
    builder = builder.subject_name(subject)
    builder = builder.issuer_name(issuer)

    builder = builder.not_valid_before(now)
    builder = builder.not_valid_after(now + expiration_delta)
    builder = builder.serial_number(int(uuid.uuid4()))
    builder = builder.public_key(root_public_key)
    builder = builder.add_extension(
        x509.BasicConstraints(ca=True, path_length=None), critical=True,
    )

    # Sign the CSR
    certificate = builder.sign(
        private_key=root_private_key, algorithm=hashes.SHA256(),
        backend=default_backend()
    )

    return root_private_key, certificate


def generate_signing_csr():
    """Generate signing CSR."""
    # Generate private key
    signing_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=3072,
        backend=default_backend()
    )

    builder = x509.CertificateSigningRequestBuilder()
    subject = x509.Name([
        x509.NameAttribute(NameOID.DOMAIN_COMPONENT, u'org'),
        x509.NameAttribute(NameOID.DOMAIN_COMPONENT, u'simple'),
        x509.NameAttribute(NameOID.COMMON_NAME, u'Simple Signing CA'),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u'Simple Inc'),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, u'Simple Signing CA'),
    ])
    builder = builder.subject_name(subject)
    builder = builder.add_extension(
        x509.BasicConstraints(ca=True, path_length=None), critical=True,
    )

    # Sign the CSR
    csr = builder.sign(
        private_key=signing_private_key, algorithm=hashes.SHA256(),
        backend=default_backend()
    )

    return signing_private_key, csr


def sign_certificate(csr, issuer_private_key, issuer_name, days_to_expiration=365, ca=False):
    """
    Sign the incoming CSR request.

    Args:
        csr                : Certificate Signing Request object
        issuer_private_key : Root CA private key if the request is for the signing
                             CA; Signing CA private key otherwise
        issuer_name        : x509 Name
        days_to_expiration : int (365 days by default)
        ca                 : Is this a certificate authority
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
        x509.BasicConstraints(ca=ca, path_length=None), critical=True,
    )

    signed_cert = builder.sign(
        private_key=issuer_private_key, algorithm=hashes.SHA256(), backend=default_backend()
    )
    return signed_cert
