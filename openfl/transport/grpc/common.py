# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Common functions for gRPC transport."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc

from openfl.protocols import aggregator_pb2

logger = logging.getLogger(__name__)


MAX_METADATA_SIZE_BYTES = 32 * 2**20
MAX_MESSAGE_LENGTH_BYTES = 2**30

DEFAULT_CHANNEL_OPTIONS = [
    ("grpc.max_metadata_size", MAX_METADATA_SIZE_BYTES),
    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH_BYTES),
]


def synchronized(func):
    """Executes `func` synchronously in a threading lock."""
    _lock = threading.Lock()

    def wrapper(self, *args, **kwargs):
        with _lock:
            return func(self, *args, **kwargs)

    return wrapper


def create_insecure_channel(uri) -> grpc.Channel:
    """Creates an insecure gRPC channel."""
    return grpc.insecure_channel(uri, options=DEFAULT_CHANNEL_OPTIONS)


def create_tls_channel(
    uri, root_certificate, require_client_auth, certificate, private_key
) -> grpc.Channel:
    """
    Creates a TLS-based gRPC channel.

    Args:
        uri (str): The uniform resource identifier for the secure channel.
        root_certificate (str): The Certificate Authority filename.
        require_client_auth (bool): True enables client-side
            authentication, i.e. mTLS.
        certificate (str): The client certificate filename from the
            collaborator (signed by the certificate authority).
        private_key (str): The private key filename for the client
            certificate.

    Returns:
        grpc.Channel: A secure gRPC channel object
    """
    with open(root_certificate, "rb") as f:
        root_certificate_b = f.read()

    if not require_client_auth:
        logger.warning("Client-side authentication is disabled.")
        private_key_b = None
        certificate_b = None
    else:
        with open(private_key, "rb") as f:
            private_key_b = f.read()
        with open(certificate, "rb") as f:
            certificate_b = f.read()

    credentials = grpc.ssl_channel_credentials(
        root_certificates=root_certificate_b,
        private_key=private_key_b,
        certificate_chain=certificate_b,
    )

    return grpc.secure_channel(uri, credentials, options=DEFAULT_CHANNEL_OPTIONS)


def create_grpc_server(
    uri: str,
    use_tls: bool,
    private_key: str,
    certificate: str,
    root_certificate: str,
    require_client_auth: bool,
    root_certificate_refresher_cb=None,
) -> grpc.server:
    """Creates a gRPC server.

    Args:
        uri (str): Address for the server (including port number).
        use_tls (bool): If set, the server will use TLS.
        private_key (str): Path to the private key file, required when TLS is enabled.
        certificate (str): Path to the certificate file, required when TLS is enabled.
        root_certificate (str): Path to the root certificate file, required when TLS is enabled.
        require_client_auth (bool): If set, client-side authentication is mandated.
            Only valid when TLS is enabled.
        root_certificate_refresher_cb (function): Callback function to refresh the root certificate.

    Returns:
        grpc.server instance.
    """
    server = grpc.server(ThreadPoolExecutor(max_workers=8), options=DEFAULT_CHANNEL_OPTIONS)

    if not use_tls:
        logger.warning("gRPC is running on insecure channel with TLS disabled.")
        port = server.add_insecure_port(uri)
        logger.info("Insecure port: %s", port)

    else:
        with open(private_key, "rb") as f:
            private_key_b = f.read()
        with open(certificate, "rb") as f:
            certificate_b = f.read()
        with open(root_certificate, "rb") as f:
            root_certificate_b = f.read()

        cert_config = grpc.ssl_server_certificate_configuration(
            ((private_key_b, certificate_b),), root_certificates=root_certificate_b
        )

        def certificate_configuration_fetcher():
            root_cert = root_certificate_b
            if root_certificate_refresher_cb:
                root_cert = root_certificate_refresher_cb()
            return grpc.ssl_server_certificate_configuration(
                ((private_key_b, certificate_b),), root_certificates=root_cert
            )

        if not require_client_auth:
            logger.warning("Client-side authentication is disabled.")
        server_credentials = grpc.dynamic_ssl_server_credentials(
            cert_config,
            certificate_configuration_fetcher,
            require_client_authentication=require_client_auth,
        )
        server.add_secure_port(uri, server_credentials)

    return server


def create_header(sender, receiver, federation_uuid, single_col_cert_common_name):
    """Creates a header for gRPC messages.

    Args:
        sender (str): The sender of the message.
        receiver (str): The receiver of the message.
        federation_uuid (str): The UUID of the federation.
        single_col_cert_common_name (str): The common name on the collaborator's certificate.

    Returns:
        aggregator_pb2.MessageHeader: The header for gRPC messages.
    """
    return aggregator_pb2.MessageHeader(
        sender=sender,
        receiver=receiver,
        federation_uuid=federation_uuid,
        single_col_cert_common_name=single_col_cert_common_name,
    )
