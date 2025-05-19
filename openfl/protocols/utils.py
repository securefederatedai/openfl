# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Proto utils."""

import logging

from openfl.protocols import base_pb2
from openfl.utilities import TensorKey

logger = logging.getLogger(__name__)


def model_proto_to_bytes_and_metadata(model_proto):
    """Convert the model protobuf to bytes and metadata.

    Args:
        model_proto: The protobuf of the model.

    Returns:
        bytes_dict: A dictionary where the keys are tensor names and the
            values are the corresponding tensor data in bytes.
        metadata_dict: A dictionary where the keys are tensor names and the
            values are the corresponding tensor metadata.
        round_number: The round number for the model.
    """
    bytes_dict = {}
    metadata_dict = {}
    round_number = None
    for tensor_proto in model_proto.tensors:
        bytes_dict[tensor_proto.name] = tensor_proto.data_bytes
        metadata_dict[tensor_proto.name] = [
            {
                "int_to_float": proto.int_to_float,
                "int_list": proto.int_list,
                "bool_list": proto.bool_list,
            }
            for proto in tensor_proto.transformer_metadata
        ]
        if round_number is None:
            round_number = tensor_proto.round_number
        else:
            assert round_number == tensor_proto.round_number, (
                f"Round numbers in model are inconsistent: {round_number} "
                f"and {tensor_proto.round_number}"
            )
    return bytes_dict, metadata_dict, round_number


def bytes_and_metadata_to_model_proto(bytes_dict, model_id, model_version, is_delta, metadata_dict):
    """
    Convert bytes and metadata to model protobuf.

    Args:
        bytes_dict: A dictionary where the keys are tensor names and the
            values are the corresponding tensor data in bytes.
        model_id: The ID of the model.
        model_version: The version of the model.
        is_delta: A flag indicating whether the model is a delta model.
        metadata_dict: A dictionary where the keys are tensor names and the
            values are the corresponding tensor metadata.

    Returns:
        model_proto: The protobuf of the model.
    """
    model_header = ModelHeader(id=model_id, version=model_version, is_delta=is_delta)  # noqa: F821

    tensor_protos = []
    for key, data_bytes in bytes_dict.items():
        transformer_metadata = metadata_dict[key]
        metadata_protos = []
        for metadata in transformer_metadata:
            if metadata.get("int_to_float") is not None:
                int_to_float = metadata.get("int_to_float")
            else:
                int_to_float = {}

            if metadata.get("int_list") is not None:
                int_list = metadata.get("int_list")
            else:
                int_list = []

            if metadata.get("bool_list") is not None:
                bool_list = metadata.get("bool_list")
            else:
                bool_list = []
            metadata_protos.append(
                base_pb2.MetadataProto(
                    int_to_float=int_to_float,
                    int_list=int_list,
                    bool_list=bool_list,
                )
            )
        tensor_protos.append(
            TensorProto(  # noqa: F821
                name=key,
                data_bytes=data_bytes,
                transformer_metadata=metadata_protos,
            )
        )
    return base_pb2.ModelProto(header=model_header, tensors=tensor_protos)


def construct_named_tensor(tensor_key, nparray, transformer_metadata, lossless):
    """Construct named tensor.

    Args:
        tensor_key: The key of the tensor.
        nparray: The numpy array representing the tensor data.
        transformer_metadata: The transformer metadata for the tensor.
        lossless: A flag indicating whether the tensor is lossless.

    Returns:
        named_tensor: The named tensor.
    """
    metadata_protos = []
    for metadata in transformer_metadata:
        if metadata.get("int_to_float") is not None:
            int_to_float = metadata.get("int_to_float")
        else:
            int_to_float = {}

        if metadata.get("int_list") is not None:
            int_list = metadata.get("int_list")
        else:
            int_list = []

        if metadata.get("bool_list") is not None:
            bool_list = metadata.get("bool_list")
        else:
            bool_list = []
        metadata_protos.append(
            base_pb2.MetadataProto(
                int_to_float=int_to_float,
                int_list=int_list,
                bool_list=bool_list,
            )
        )

    tensor_name, origin, round_number, report, tags = tensor_key

    return base_pb2.NamedTensor(
        name=tensor_name,
        round_number=round_number,
        lossless=lossless,
        report=report,
        tags=tags,
        transformer_metadata=metadata_protos,
        data_bytes=nparray,
    )


def construct_proto(tensor_dict, model_id, model_version, is_delta, compression_pipeline):
    """Construct proto.

    Args:
        tensor_dict: A dictionary where the keys are tensor names and the
            values are the corresponding tensors.
        model_id: The ID of the model.
        model_version: The version of the model.
        is_delta: A flag indicating whether the model is a delta model.
        compression_pipeline: The compression pipeline for the model.

    Returns:
        model_proto: The protobuf of the model.
    """
    # compress the arrays in the tensor_dict, and form the model proto
    # TODO: Hold-out tensors from the compression pipeline.
    bytes_dict = {}
    metadata_dict = {}
    for key, array in tensor_dict.items():
        bytes_dict[key], metadata_dict[key] = compression_pipeline.forward(data=array)

    # convert the compressed_tensor_dict and metadata to protobuf, and make the new model proto
    model_proto = bytes_and_metadata_to_model_proto(
        bytes_dict=bytes_dict,
        model_id=model_id,
        model_version=model_version,
        is_delta=is_delta,
        metadata_dict=metadata_dict,
    )
    return model_proto


def construct_model_proto(tensor_dict, round_number, tensor_pipe):
    """Construct model proto from tensor dict.

    Args:
        tensor_dict: A dictionary where the keys are tensor names and the
            values are the corresponding tensors.
        round_number: The round number for the model.
        tensor_pipe: The tensor pipe for the model.

    Returns:
        model_proto: The protobuf of the model.
    """
    # compress the arrays in the tensor_dict, and form the model proto
    # TODO: Hold-out tensors from the tensor compression pipeline.
    named_tensors = []
    for key, nparray in tensor_dict.items():
        bytes_data, transformer_metadata = tensor_pipe.forward(data=nparray)
        tensor_key = TensorKey(key, "agg", round_number, False, ("model",))
        named_tensors.append(
            construct_named_tensor(
                tensor_key,
                bytes_data,
                transformer_metadata,
                lossless=True,
            )
        )

    return base_pb2.ModelProto(tensors=named_tensors)


def deconstruct_model_proto(model_proto, compression_pipeline):
    """Deconstruct model proto.

    This function takes a model protobuf and a compression pipeline,
    and deconstructs the protobuf into a dictionary of tensors and a round
    number.

    Args:
        model_proto: The protobuf of the model.
        compression_pipeline: The compression pipeline for the model.

    Returns:
        tensor_dict: A dictionary where the keys are tensor names and the
        values are the corresponding tensors.
        round_number: The round number for the model.
    """
    # extract the tensor_dict and metadata
    bytes_dict, metadata_dict, round_number = model_proto_to_bytes_and_metadata(model_proto)

    # decompress the tensors
    # TODO: Handle tensors meant to be held-out from the compression pipeline
    #  (currently none are held out).
    tensor_dict = {}
    for key in bytes_dict:
        tensor_dict[key] = compression_pipeline.backward(
            data=bytes_dict[key], transformer_metadata=metadata_dict[key]
        )
    return tensor_dict, round_number


def deconstruct_proto(model_proto, compression_pipeline):
    """Deconstruct the protobuf.

    This function takes a model protobuf and a compression pipeline, and
    deconstructs the protobuf into a dictionary of tensors.

    Args:
        model_proto: The protobuf of the model.
        compression_pipeline: The compression pipeline for the model.

    Returns:
        tensor_dict: A dictionary where the keys are tensor names and the
            values are the corresponding tensors.
    """
    # extract the tensor_dict and metadata
    bytes_dict, metadata_dict = model_proto_to_bytes_and_metadata(model_proto)

    # decompress the tensors
    # TODO: Handle tensors meant to be held-out from the compression pipeline
    #  (currently none are held out).
    tensor_dict = {}
    for key in bytes_dict:
        tensor_dict[key] = compression_pipeline.backward(
            data=bytes_dict[key], transformer_metadata=metadata_dict[key]
        )
    return tensor_dict


def load_proto(fpath):
    """Load the protobuf.

    Args:
        fpath: The file path of the protobuf.

    Returns:
        model: The protobuf of the model.
    """
    with open(fpath, "rb") as f:
        loaded = f.read()
        model = base_pb2.ModelProto().FromString(loaded)
        return model


def dump_proto(model_proto, fpath):
    """Dump the protobuf to a file.

    Args:
        model_proto: The protobuf of the model.
        fpath: The file path to dump the protobuf.
    """
    s = model_proto.SerializeToString()
    with open(fpath, "wb") as f:
        f.write(s)


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
        reply = base_pb2.DataStream(npbytes=chunk, size=len(chunk))
        yield reply


def get_headers(context) -> dict:
    """Get headers from context.

    Args:
        context: The context containing the headers.

    Returns:
        headers: A dictionary where the keys are header names and the
            values are the corresponding header values.
    """
    return {header[0]: header[1] for header in context.invocation_metadata()}
