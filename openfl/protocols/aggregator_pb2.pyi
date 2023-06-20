from openfl.protocols import base_pb2 as _base_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetAggregatedTensorRequest(_message.Message):
    __slots__ = ["header", "report", "require_lossless", "round_number", "tags", "tensor_name"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    REQUIRE_LOSSLESS_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    TENSOR_NAME_FIELD_NUMBER: _ClassVar[int]
    header: MessageHeader
    report: bool
    require_lossless: bool
    round_number: int
    tags: _containers.RepeatedScalarFieldContainer[str]
    tensor_name: str
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ..., tensor_name: _Optional[str] = ..., round_number: _Optional[int] = ..., report: bool = ..., tags: _Optional[_Iterable[str]] = ..., require_lossless: bool = ...) -> None: ...

class GetAggregatedTensorResponse(_message.Message):
    __slots__ = ["header", "round_number", "tensor"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    header: MessageHeader
    round_number: int
    tensor: _base_pb2.NamedTensor
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ..., round_number: _Optional[int] = ..., tensor: _Optional[_Union[_base_pb2.NamedTensor, _Mapping]] = ...) -> None: ...

class GetExperimentDescriptionRequest(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetExperimentDescriptionResponse(_message.Message):
    __slots__ = ["experiment"]
    EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
    experiment: _base_pb2.ExperimentDescription
    def __init__(self, experiment: _Optional[_Union[_base_pb2.ExperimentDescription, _Mapping]] = ...) -> None: ...

class GetMetricStreamRequest(_message.Message):
    __slots__ = ["experiment_name"]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_name: str
    def __init__(self, experiment_name: _Optional[str] = ...) -> None: ...

class GetMetricStreamResponse(_message.Message):
    __slots__ = ["metric_name", "metric_origin", "metric_value", "round", "task_name"]
    METRIC_NAME_FIELD_NUMBER: _ClassVar[int]
    METRIC_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    METRIC_VALUE_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    TASK_NAME_FIELD_NUMBER: _ClassVar[int]
    metric_name: str
    metric_origin: str
    metric_value: float
    round: int
    task_name: str
    def __init__(self, metric_origin: _Optional[str] = ..., task_name: _Optional[str] = ..., metric_name: _Optional[str] = ..., metric_value: _Optional[float] = ..., round: _Optional[int] = ...) -> None: ...

class GetTasksRequest(_message.Message):
    __slots__ = ["header"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    header: MessageHeader
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ...) -> None: ...

class GetTasksResponse(_message.Message):
    __slots__ = ["header", "quit", "round_number", "sleep_time", "tasks"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    QUIT_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SLEEP_TIME_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    header: MessageHeader
    quit: bool
    round_number: int
    sleep_time: int
    tasks: _containers.RepeatedCompositeFieldContainer[Task]
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ..., round_number: _Optional[int] = ..., tasks: _Optional[_Iterable[_Union[Task, _Mapping]]] = ..., sleep_time: _Optional[int] = ..., quit: bool = ...) -> None: ...

class GetTrainedModelRequest(_message.Message):
    __slots__ = ["experiment_name", "model_type"]
    class ModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BEST_MODEL: GetTrainedModelRequest.ModelType
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    LAST_MODEL: GetTrainedModelRequest.ModelType
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    experiment_name: str
    model_type: GetTrainedModelRequest.ModelType
    def __init__(self, experiment_name: _Optional[str] = ..., model_type: _Optional[_Union[GetTrainedModelRequest.ModelType, str]] = ...) -> None: ...

class MessageHeader(_message.Message):
    __slots__ = ["federation_uuid", "receiver", "sender", "single_col_cert_common_name"]
    FEDERATION_UUID_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_FIELD_NUMBER: _ClassVar[int]
    SENDER_FIELD_NUMBER: _ClassVar[int]
    SINGLE_COL_CERT_COMMON_NAME_FIELD_NUMBER: _ClassVar[int]
    federation_uuid: str
    receiver: str
    sender: str
    single_col_cert_common_name: str
    def __init__(self, sender: _Optional[str] = ..., receiver: _Optional[str] = ..., federation_uuid: _Optional[str] = ..., single_col_cert_common_name: _Optional[str] = ...) -> None: ...

class SendLocalTaskResultsResponse(_message.Message):
    __slots__ = ["header"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    header: MessageHeader
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ...) -> None: ...

class Task(_message.Message):
    __slots__ = ["apply_local", "function_name", "name", "task_type"]
    APPLY_LOCAL_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_TYPE_FIELD_NUMBER: _ClassVar[int]
    apply_local: bool
    function_name: str
    name: str
    task_type: str
    def __init__(self, name: _Optional[str] = ..., function_name: _Optional[str] = ..., task_type: _Optional[str] = ..., apply_local: bool = ...) -> None: ...

class TaskResults(_message.Message):
    __slots__ = ["data_size", "header", "round_number", "task_name", "tensors"]
    DATA_SIZE_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TASK_NAME_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    data_size: int
    header: MessageHeader
    round_number: int
    task_name: str
    tensors: _containers.RepeatedCompositeFieldContainer[_base_pb2.NamedTensor]
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ..., round_number: _Optional[int] = ..., task_name: _Optional[str] = ..., data_size: _Optional[int] = ..., tensors: _Optional[_Iterable[_Union[_base_pb2.NamedTensor, _Mapping]]] = ...) -> None: ...

class TrainedModelResponse(_message.Message):
    __slots__ = ["model_proto"]
    MODEL_PROTO_FIELD_NUMBER: _ClassVar[int]
    model_proto: _base_pb2.ModelProto
    def __init__(self, model_proto: _Optional[_Union[_base_pb2.ModelProto, _Mapping]] = ...) -> None: ...
