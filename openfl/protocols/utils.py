# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.utilities import TensorKey
from openfl.protocols import ModelProto, NamedTensor, MetadataProto, DataStream


def model_proto_to_bytes_and_metadata(model_proto):
    """Converts the model protobuf to bytes and metadata

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
        metadata_dict[tensor_proto.name] = [{'int_to_float': proto.int_to_float,
                                             'int_list': proto.int_list,
                                             'bool_list': proto.bool_list} for proto in
                                            tensor_proto.transformer_metadata]
        if round_number is None:
            round_number = tensor_proto.round_number
        else:
            assert round_number == tensor_proto.round_number, (
                'Round numbers in model are inconsistent: {} and {}'.format(
                    round_number, tensor_proto.round_number))
    return bytes_dict, metadata_dict, round_number


def bytes_and_metadata_to_model_proto(bytes_dict, model_id, model_version, is_delta, metadata_dict):
    model_header = ModelHeader(id=model_id, version=model_version, is_delta=is_delta)

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
            metadata_protos.append(MetadataProto(int_to_float=int_to_float, int_list=int_list, bool_list=bool_list))
        tensor_protos.append(TensorProto(name=key,
                                         data_bytes=data_bytes,
                                         transformer_metadata=metadata_protos))
    return ModelProto(header=model_header, tensors=tensor_protos)


def construct_named_tensor(tensor_key, nparray, transformer_metadata, lossless):
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
        metadata_protos.append(MetadataProto(int_to_float=int_to_float, int_list=int_list, bool_list=bool_list))

    tensor_name, origin, round_number, report, tags = tensor_key

    return NamedTensor(name=tensor_name, round_number=round_number, lossless=lossless, report=report, tags=tags,
                       transformer_metadata=metadata_protos, data_bytes=nparray)


def construct_proto(tensor_dict, model_id, model_version, is_delta, compression_pipeline):
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
    # compress the arrays in the tensor_dict, and form the model proto
    # TODO: Hold-out tensors from the tensor compression pipeline.
    named_tensors = []
    for key, nparray in tensor_dict.items():
        bytes, transformer_metadata = tensor_pipe.forward(data=nparray)
        tensor_key = TensorKey(key, 'agg', round_number, False, ('model',))
        named_tensors.append(construct_named_tensor(tensor_key, bytes, transformer_metadata, lossless=True))

    return ModelProto(tensors=named_tensors)


def deconstruct_model_proto(model_proto, compression_pipeline):
    # extract the tensor_dict and metadata
    bytes_dict, metadata_dict, round_number = model_proto_to_bytes_and_metadata(model_proto)

    # decompress the tensors
    # TODO: Handle tensors meant to be held-out from the compression pipeline (currently none are held out).
    tensor_dict = {}
    for key in bytes_dict:
        tensor_dict[key] = compression_pipeline.backward(data=bytes_dict[key],
                                                         transformer_metadata=metadata_dict[key])
    return tensor_dict, round_number


def deconstruct_proto(model_proto, compression_pipeline):
    """Deconstruct the protobuf

    Args:
        model_proto: The protobuf of the model
        compression_pipeline: The compression pipeline object

    Returns:
        protobuf: A protobuf of the model
    """

    # extract the tensor_dict and metadata
    bytes_dict, metadata_dict = model_proto_to_bytes_and_metadata(model_proto)

    # decompress the tensors
    # TODO: Handle tensors meant to be held-out from the compression pipeline (currently none are held out).
    tensor_dict = {}
    for key in bytes_dict:
        tensor_dict[key] = compression_pipeline.backward(data=bytes_dict[key],
                                                         transformer_metadata=metadata_dict[key])
    return tensor_dict


def load_proto(fpath):
    """Load the protobuf

    Args:
        fpath: The filepath for the protobuf

    Returns:
        protobuf: A protobuf of the model
    """
    with open(fpath, "rb") as f:
        loaded = f.read()
        model = ModelProto().FromString(loaded)
        return model


def dump_proto(model_proto, fpath):
    """Dumps the protobuf to a file

    Args:
        model_proto: The protobuf of the model
        fpath: The filename to save the model protobuf

    """
    s = model_proto.SerializeToString()
    with open(fpath, "wb") as f:
        f.write(s)


def datastream_to_proto(proto, stream, logger=None):
    """Converts the datastream to the protobuf

    Args:
        model_proto: The protobuf of the model
        stream: The data stream from the remote connection
        logger: (Optional) The log object

    Returns:
        protobuf: A protobuf of the model
    """
    npbytes = b""
    for chunk in stream:
        npbytes += chunk.npbytes

    if len(npbytes) > 0:
        proto.ParseFromString(npbytes)
        if logger is not None:
            logger.debug("datastream_to_proto parsed a {}.".format(type(proto)))
        return proto
    else:
        raise RuntimeError("Received empty stream message of type {}".format(type(proto)))


def proto_to_datastream(proto, logger, max_buffer_size=(2 * 1024 * 1024)):
    """Convert the protobuf to the datastream for the remote connection

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
    logger.debug("Setting stream chunks with size {} for proto of type {}".format(buffer_size, type(proto)))

    for i in range(0, data_size, buffer_size):
        chunk = npbytes[i: i + buffer_size]
        reply = DataStream(npbytes=chunk, size=len(chunk))
        yield reply
