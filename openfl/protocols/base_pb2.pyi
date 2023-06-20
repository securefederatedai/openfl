from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CollaboratorDescription(_message.Message):
    __slots__ = ["current_task", "name", "next_task", "progress", "round", "status"]
    CURRENT_TASK_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NEXT_TASK_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    current_task: str
    name: str
    next_task: str
    progress: float
    round: int
    status: str
    def __init__(self, name: _Optional[str] = ..., status: _Optional[str] = ..., progress: _Optional[float] = ..., round: _Optional[int] = ..., current_task: _Optional[str] = ..., next_task: _Optional[str] = ...) -> None: ...

class DataStream(_message.Message):
    __slots__ = ["npbytes", "size"]
    NPBYTES_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    npbytes: bytes
    size: int
    def __init__(self, size: _Optional[int] = ..., npbytes: _Optional[bytes] = ...) -> None: ...

class DownloadStatus(_message.Message):
    __slots__ = ["name", "status"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    name: str
    status: str
    def __init__(self, name: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class DownloadStatuses(_message.Message):
    __slots__ = ["logs", "models"]
    LOGS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    logs: _containers.RepeatedCompositeFieldContainer[DownloadStatus]
    models: _containers.RepeatedCompositeFieldContainer[DownloadStatus]
    def __init__(self, models: _Optional[_Iterable[_Union[DownloadStatus, _Mapping]]] = ..., logs: _Optional[_Iterable[_Union[DownloadStatus, _Mapping]]] = ...) -> None: ...

class ExperimentDescription(_message.Message):
    __slots__ = ["collaborators", "current_round", "download_statuses", "name", "progress", "status", "tasks", "total_rounds"]
    COLLABORATORS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_STATUSES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    collaborators: _containers.RepeatedCompositeFieldContainer[CollaboratorDescription]
    current_round: int
    download_statuses: DownloadStatuses
    name: str
    progress: float
    status: str
    tasks: _containers.RepeatedCompositeFieldContainer[TaskDescription]
    total_rounds: int
    def __init__(self, name: _Optional[str] = ..., status: _Optional[str] = ..., progress: _Optional[float] = ..., total_rounds: _Optional[int] = ..., current_round: _Optional[int] = ..., download_statuses: _Optional[_Union[DownloadStatuses, _Mapping]] = ..., collaborators: _Optional[_Iterable[_Union[CollaboratorDescription, _Mapping]]] = ..., tasks: _Optional[_Iterable[_Union[TaskDescription, _Mapping]]] = ...) -> None: ...

class MetadataProto(_message.Message):
    __slots__ = ["bool_list", "int_list", "int_to_float"]
    class IntToFloatEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(self, key: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...
    BOOL_LIST_FIELD_NUMBER: _ClassVar[int]
    INT_LIST_FIELD_NUMBER: _ClassVar[int]
    INT_TO_FLOAT_FIELD_NUMBER: _ClassVar[int]
    bool_list: _containers.RepeatedScalarFieldContainer[bool]
    int_list: _containers.RepeatedScalarFieldContainer[int]
    int_to_float: _containers.ScalarMap[int, float]
    def __init__(self, int_to_float: _Optional[_Mapping[int, float]] = ..., int_list: _Optional[_Iterable[int]] = ..., bool_list: _Optional[_Iterable[bool]] = ...) -> None: ...

class ModelProto(_message.Message):
    __slots__ = ["tensors"]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.RepeatedCompositeFieldContainer[NamedTensor]
    def __init__(self, tensors: _Optional[_Iterable[_Union[NamedTensor, _Mapping]]] = ...) -> None: ...

class NamedTensor(_message.Message):
    __slots__ = ["data_bytes", "lossless", "name", "report", "round_number", "tags", "transformer_metadata"]
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    LOSSLESS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    TRANSFORMER_METADATA_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    lossless: bool
    name: str
    report: bool
    round_number: int
    tags: _containers.RepeatedScalarFieldContainer[str]
    transformer_metadata: _containers.RepeatedCompositeFieldContainer[MetadataProto]
    def __init__(self, name: _Optional[str] = ..., round_number: _Optional[int] = ..., lossless: bool = ..., report: bool = ..., tags: _Optional[_Iterable[str]] = ..., transformer_metadata: _Optional[_Iterable[_Union[MetadataProto, _Mapping]]] = ..., data_bytes: _Optional[bytes] = ...) -> None: ...

class TaskDescription(_message.Message):
    __slots__ = ["description", "name"]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    description: str
    name: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...
