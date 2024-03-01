# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Proto utils."""

from openfl.protocols import base_pb2
from openfl.utilities import TensorKey


def model_proto_to_bytes_and_metadata(model_proto):
    """Convert the model protobuf to bytes and metadata.

    Args:
        model_proto: Protobuf of the model

    Returns:
        bytes_dict: Dictionary of the bytes contained in the model protobuf
        metadata_dict: Dictionary of the meta data in the model protobuf
    """
    bytes_dict = {}
    metadata_dict = {}
    round_number = None
    for tensor_proto in model_proto.tensors:
        bytes_dict[tensor_proto.name] = tensor_proto.data_bytes
        metadata_dict[tensor_proto.name] = [{
            'int_to_float': proto.int_to_float,
            'int_list': proto.int_list,
            'bool_list': proto.bool_list
        }
            for proto in tensor_proto.transformer_metadata
        ]
        if round_number is None:
            round_number = tensor_proto.round_number
        else:
            assert round_number == tensor_proto.round_number, (
                f'Round numbers in model are inconsistent: {round_number} '
                f'and {tensor_proto.round_number}'
            )
    return bytes_dict, metadata_dict, round_number


def bytes_and_metadata_to_model_proto(bytes_dict, model_id, model_version,
                                      is_delta, metadata_dict):
    """Convert bytes and metadata to model protobuf."""
    model_header = ModelHeader(id=model_id, version=model_version, is_delta=is_delta)  # NOQA:F821

    tensor_protos = []
    for key, data_bytes in bytes_dict.items():
        transformer_metadata = metadata_dict[key]
        metadata_protos = []
        for metadata in transformer_metadata:
            if metadata.get('int_to_float') is not None:
                int_to_float = metadata.get('int_to_float')
            else:
                int_to_float = {}

            if metadata.get('int_list') is not None:
                int_list = metadata.get('int_list')
            else:
                int_list = []

            if metadata.get('bool_list') is not None:
                bool_list = metadata.get('bool_list')
            else:
                bool_list = []
            metadata_protos.append(base_pb2.MetadataProto(
                int_to_float=int_to_float,
                int_list=int_list,
                bool_list=bool_list,
            ))
        tensor_protos.append(TensorProto(name=key,  # NOQA:F821
                                         data_bytes=data_bytes,
                                         transformer_metadata=metadata_protos))
    return base_pb2.ModelProto(header=model_header, tensors=tensor_protos)


def construct_named_tensor(tensor_key, nparray, transformer_metadata, lossless):
    """Construct named tensor."""
    metadata_protos = []
    for metadata in transformer_metadata:
        if metadata.get('int_to_float') is not None:
            int_to_float = metadata.get('int_to_float')
        else:
            int_to_float = {}

        if metadata.get('int_list') is not None:
            int_list = metadata.get('int_list')
        else:
            int_list = []

        if metadata.get('bool_list') is not None:
            bool_list = metadata.get('bool_list')
        else:
            bool_list = []
        metadata_protos.append(base_pb2.MetadataProto(
            int_to_float=int_to_float,
            int_list=int_list,
            bool_list=bool_list,
        ))

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
    """Construct proto."""
    # compress the arrays in the tensor_dict, and form the model proto
    # TODO: Hold-out tensors from the compression pipeline.
    bytes_dict = {}
    metadata_dict = {}
    for key, array in tensor_dict.items():
        bytes_dict[key], metadata_dict[key] = compression_pipeline.forward(data=array)

    # convert the compressed_tensor_dict and metadata to protobuf, and make the new model proto
    model_proto = bytes_and_metadata_to_model_proto(bytes_dict=bytes_dict,
                                                    model_id=model_id,
                                                    model_version=model_version,
                                                    is_delta=is_delta,
                                                    metadata_dict=metadata_dict)
    return model_proto


def construct_model_proto(tensor_dict, round_number, tensor_pipe):
    """Construct model proto from tensor dict."""
    # compress the arrays in the tensor_dict, and form the model proto
    # TODO: Hold-out tensors from the tensor compression pipeline.
    named_tensors = []
    for key, nparray in tensor_dict.items():
        bytes_data, transformer_metadata = tensor_pipe.forward(data=nparray)
        tensor_key = TensorKey(key, 'agg', round_number, False, ('model',))
        named_tensors.append(construct_named_tensor(
            tensor_key,
            bytes_data,
            transformer_metadata,
            lossless=True,
        ))

    return base_pb2.ModelProto(tensors=named_tensors)


def deconstruct_model_proto(model_proto, compression_pipeline):
    """Deconstruct model proto."""
    # extract the tensor_dict and metadata
    bytes_dict, metadata_dict, round_number = model_proto_to_bytes_and_metadata(model_proto)

    # decompress the tensors
    # TODO: Handle tensors meant to be held-out from the compression pipeline
    #  (currently none are held out).
    tensor_dict = {}
    for key in bytes_dict:
        tensor_dict[key] = compression_pipeline.backward(data=bytes_dict[key],
                                                         transformer_metadata=metadata_dict[key])
    return tensor_dict, round_number


def deconstruct_proto(model_proto, compression_pipeline):
    """Deconstruct the protobuf.

    Args:
        model_proto: The protobuf of the model
        compression_pipeline: The compression pipeline object

    Returns:
        protobuf: A protobuf of the model
    """
    # extract the tensor_dict and metadata
    bytes_dict, metadata_dict = model_proto_to_bytes_and_metadata(model_proto)

    # decompress the tensors
    # TODO: Handle tensors meant to be held-out from the compression pipeline
    #  (currently none are held out).
    tensor_dict = {}
    for key in bytes_dict:
        tensor_dict[key] = compression_pipeline.backward(data=bytes_dict[key],
                                                         transformer_metadata=metadata_dict[key])
    return tensor_dict


def load_proto(fpath):
    """Load the protobuf.

    Args:
        fpath: The filepath for the protobuf

    Returns:
        protobuf: A protobuf of the model
    """
    with open(fpath, 'rb') as f:
        loaded = f.read()
        model = base_pb2.ModelProto().FromString(loaded)
        return model


def dump_proto(model_proto, fpath):
    """Dump the protobuf to a file.

    Args:
        model_proto: The protobuf of the model
        fpath: The filename to save the model protobuf

    """
    s = model_proto.SerializeToString()
    with open(fpath, 'wb') as f:
        f.write(s)


def datastream_to_proto(proto, stream, logger=None):
    """Convert the datastream to the protobuf.

    Args:
        model_proto: The protobuf of the model
        stream: The data stream from the remote connection
        logger: (Optional) The log object

    Returns:
        protobuf: A protobuf of the model
    """
    npbytes = b''
    for chunk in stream:
        npbytes += chunk.npbytes

    if len(npbytes) > 0:
        proto.ParseFromString(npbytes)
        if logger is not None:
            logger.debug(f'datastream_to_proto parsed a {type(proto)}.')
        return proto
    else:
        raise RuntimeError(f'Received empty stream message of type {type(proto)}')


def proto_to_datastream(proto, logger, max_buffer_size=(2 * 1024 * 1024)):
    """Convert the protobuf to the datastream for the remote connection.

    Args:
        model_proto: The protobuf of the model
        logger: The log object
        max_buffer_size: The buffer size (Default= 2*1024*1024)
    Returns:
        reply: The message for the remote connection.
    """
    npbytes = proto.SerializeToString()
    data_size = len(npbytes)
    buffer_size = data_size if max_buffer_size > data_size else max_buffer_size
    logger.debug(f'Setting stream chunks with size {buffer_size} for proto of type {type(proto)}')

    for i in range(0, data_size, buffer_size):
        chunk = npbytes[i: i + buffer_size]
        reply = base_pb2.DataStream(npbytes=chunk, size=len(chunk))
        yield reply


def get_headers(context) -> dict:
    """Get headers from context."""
    return {header[0]: header[1] for header in context.invocation_metadata()}
