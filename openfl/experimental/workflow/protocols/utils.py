# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Proto utils."""

import logging

from openfl.experimental.workflow.protocols import base_pb2

logger = logging.getLogger(__name__)


def datastream_to_proto(proto, stream):
    """Convert the datastream to the protobuf.

    Args:
        proto: The protobuf to be filled with the data stream.
        stream: The data stream.
        logger (optional): The logger for logging information.

    Returns:
        proto: The protobuf filled with the data stream.
    """
    npbytes = bytearray()
    for chunk in stream:
        npbytes.extend(chunk.npbytes)

    if len(npbytes) > 0:
        proto.ParseFromString(bytes(npbytes))
        return proto
    else:
        raise RuntimeError(f"Received empty stream message of type {type(proto)}")


def proto_to_datastream(proto, max_buffer_size=(2 * 1024 * 1024)):
    """Convert the protobuf to the datastream for the remote connection.

    Args:
        proto: The protobuf to be converted into a data stream.
        logger: The logger for logging information.
        max_buffer_size (optional): The maximum buffer size for the data
            stream. Defaults to 2*1024*1024.

    Yields:
        reply: Chunks of the data stream for the remote connection.
    """
    npbytes = proto.SerializeToString()
    data_size = len(npbytes)
    buffer_size = data_size if max_buffer_size > data_size else max_buffer_size

    for i in range(0, data_size, buffer_size):
        chunk = npbytes[i : i + buffer_size]
        reply = base_pb2.ExpDataStream(npbytes=chunk, size=len(chunk))
        yield reply
