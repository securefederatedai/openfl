# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.protocols import aggregator_pb2

max_metadata_size = 32 * 2**20
max_message_length = 2**30

channel_options = [
    ("grpc.max_metadata_size", max_metadata_size),
    ("grpc.max_send_message_length", max_message_length),
    ("grpc.max_receive_message_length", max_message_length),
]


def create_header(sender, receiver, federation_uuid, single_col_cert_common_name):
    """Create a header for gRPC messages.

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
