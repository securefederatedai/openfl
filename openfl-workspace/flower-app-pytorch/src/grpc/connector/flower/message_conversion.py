from flwr.proto import grpcadapter_pb2
from openfl.protocols import aggregator_pb2

def flower_to_openfl_message(flower_message, 
                             header=None, 
                             end_experiment=False):
    """
    Convert a Flower MessageContainer to an OpenFL DropPod.

    This function takes a Flower MessageContainer and converts it into an OpenFL DropPod.
    If the input is already an OpenFL DropPod, it returns the input as-is.

    Args:
        flower_message (grpcadapter_pb2.MessageContainer or aggregator_pb2.DropPod): 
            The Flower message to be converted. It can either be a Flower MessageContainer 
            or an OpenFL DropPod.
        header (aggregator_pb2.MessageHeader, optional): 
            An optional header to be included in the OpenFL DropPod. If provided, 
            it will be copied to the DropPod's header field.

    Returns:
        aggregator_pb2.DropPod: The converted OpenFL DropPod message.
    """
    if isinstance(flower_message, aggregator_pb2.DropPod):
        # If the input is already an OpenFL message, return it as-is
        return flower_message
    else:
        # Create the OpenFL message
        openfl_message = aggregator_pb2.DropPod()
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
    Convert an OpenFL DropPod to a Flower MessageContainer.

    This function takes an OpenFL DropPod and converts it into a Flower MessageContainer.
    If the input is already a Flower MessageContainer, it returns the input as-is.

    Args:
        openfl_message (aggregator_pb2.DropPod or grpcadapter_pb2.MessageContainer): 
            The OpenFL message to be converted. It can either be an OpenFL DropPod 
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