import importlib
from google.protobuf.message import DecodeError

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
    module_name = metadata.get('grpc-message-module')
    qualname = metadata.get('grpc-message-qualname')

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