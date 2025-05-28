# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging

from flwr.proto import grpcadapter_pb2
from google.protobuf.message import DecodeError

from openfl.protocols import aggregator_pb2

logger = logging.getLogger(__name__)


def flower_to_openfl_message(flower_message, header=None, end_experiment=False):
    """
    Convert a Flower MessageContainer to an OpenFL InteropMessage.

    This function takes a Flower MessageContainer and converts it into an OpenFL InteropMessage.
    If the input is already an OpenFL InteropMessage, it returns the input as-is.

    Args:
        flower_message (grpcadapter_pb2.MessageContainer or aggregator_pb2.InteropMessage):
            The Flower message to be converted. It can either be a Flower MessageContainer
            or an OpenFL InteropMessage.
        header (aggregator_pb2.MessageHeader, optional):
            An optional header to be included in the OpenFL InteropMessage. If provided,
            it will be copied to the InteropMessage's header field.

    Returns:
        aggregator_pb2.InteropMessage: The converted OpenFL InteropMessage message.
    """
    if isinstance(flower_message, aggregator_pb2.InteropMessage):
        # If the input is already an OpenFL message, return it as-is
        return flower_message
    else:
        # Check if the Flower message can be deserialized, log a warning if not
        try:
            deserialized_message = deserialize_flower_message(flower_message)
            if deserialized_message is None:
                logger.warning("Failed to introspect Flower message.")
        except Exception as e:
            logger.warning(f"Exception during Flower message introspection: {e}")

        # Create the OpenFL message
        openfl_message = aggregator_pb2.InteropMessage()
        # Set the MessageHeader fields based on the provided sender and receiver
        if header:
            openfl_message.header.CopyFrom(header)

        # Serialize the Flower message and set it in the OpenFL message
        serialized_flower_message = flower_message.SerializeToString()
        openfl_message.message.npbytes = serialized_flower_message
        openfl_message.message.size = len(serialized_flower_message)

        # Add flag to check if experiment has ended
        openfl_message.metadata.update({"end_experiment": str(end_experiment)})
        return openfl_message


def openfl_to_flower_message(openfl_message):
    """
    Convert an OpenFL InteropMessage to a Flower MessageContainer.

    This function takes an OpenFL InteropMessage and converts it into a Flower MessageContainer.
    If the input is already a Flower MessageContainer, it returns the input as-is.

    Args:
        openfl_message (aggregator_pb2.InteropMessage or grpcadapter_pb2.MessageContainer):
            The OpenFL message to be converted. It can either be an OpenFL InteropMessage
            or a Flower MessageContainer.

    Returns:
        grpcadapter_pb2.MessageContainer: The converted Flower MessageContainer.
    """
    if isinstance(openfl_message, grpcadapter_pb2.MessageContainer):
        # If the input is already a Flower message, return it as-is
        return openfl_message
    else:
        # Deserialize the Flower message from the DataStream npbytes field
        flower_message = grpcadapter_pb2.MessageContainer()
        flower_message.ParseFromString(openfl_message.message.npbytes)
        return flower_message


def deserialize_flower_message(flower_message):
    """
    Deserialize the grpc_message_content of a Flower message using the module and class name
    specified in the metadata.

    Args:
        flower_message: The Flower message containing the metadata and binary content.

    Returns:
        The deserialized message object, or None if deserialization fails.
    """
    # Access metadata directly
    metadata = flower_message.metadata
    module_name = metadata.get("grpc-message-module")
    qualname = metadata.get("grpc-message-qualname")

    # Import the module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import module: {module_name}. Error: {e}")
        return None

    # Get the message class
    try:
        message_class = getattr(module, qualname)
    except AttributeError as e:
        print(f"Failed to get message class '{qualname}' from module '{module_name}'. Error: {e}")
        return None

    # Deserialize the content
    try:
        message = message_class.FromString(flower_message.grpc_message_content)
    except DecodeError as e:
        print(f"Failed to deserialize message content. Error: {e}")
        return None

    return message
