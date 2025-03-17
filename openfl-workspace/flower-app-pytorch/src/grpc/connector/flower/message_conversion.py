from flwr.proto import grpcadapter_pb2
from openfl.protocols import aggregator_pb2

def flower_to_openfl_message(flower_message,
                             header=None,
                             end_experiment=False):
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
