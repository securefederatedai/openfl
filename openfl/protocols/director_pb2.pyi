from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import duration_pb2 as _duration_pb2
from openfl.protocols import base_pb2 as _base_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CudaDeviceInfo(_message.Message):
    __slots__ = ["cuda_driver_version", "cuda_version", "device_utilization", "index", "memory_total", "memory_utilized", "name"]
    CUDA_DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    CUDA_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_FIELD_NUMBER: _ClassVar[int]
    MEMORY_UTILIZED_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    cuda_driver_version: str
    cuda_version: str
    device_utilization: str
    index: int
    memory_total: int
    memory_utilized: int
    name: str
    def __init__(self, index: _Optional[int] = ..., memory_total: _Optional[int] = ..., memory_utilized: _Optional[int] = ..., device_utilization: _Optional[str] = ..., cuda_driver_version: _Optional[str] = ..., cuda_version: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class EnvoyInfo(_message.Message):
    __slots__ = ["experiment_name", "is_experiment_running", "is_online", "last_updated", "shard_info", "valid_duration"]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_EXPERIMENT_RUNNING_FIELD_NUMBER: _ClassVar[int]
    IS_ONLINE_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_FIELD_NUMBER: _ClassVar[int]
    SHARD_INFO_FIELD_NUMBER: _ClassVar[int]
    VALID_DURATION_FIELD_NUMBER: _ClassVar[int]
    experiment_name: str
    is_experiment_running: bool
    is_online: bool
    last_updated: _timestamp_pb2.Timestamp
    shard_info: ShardInfo
    valid_duration: _duration_pb2.Duration
    def __init__(self, shard_info: _Optional[_Union[ShardInfo, _Mapping]] = ..., is_online: bool = ..., is_experiment_running: bool = ..., last_updated: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., valid_duration: _Optional[_Union[_duration_pb2.Duration, _Mapping]] = ..., experiment_name: _Optional[str] = ...) -> None: ...

class ExperimentData(_message.Message):
    __slots__ = ["npbytes", "size"]
    NPBYTES_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    npbytes: bytes
    size: int
    def __init__(self, size: _Optional[int] = ..., npbytes: _Optional[bytes] = ...) -> None: ...

class ExperimentInfo(_message.Message):
    __slots__ = ["collaborator_names", "experiment_data", "model_proto", "name"]
    COLLABORATOR_NAMES_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_DATA_FIELD_NUMBER: _ClassVar[int]
    MODEL_PROTO_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    collaborator_names: _containers.RepeatedScalarFieldContainer[str]
    experiment_data: ExperimentData
    model_proto: _base_pb2.ModelProto
    name: str
    def __init__(self, name: _Optional[str] = ..., collaborator_names: _Optional[_Iterable[str]] = ..., experiment_data: _Optional[_Union[ExperimentData, _Mapping]] = ..., model_proto: _Optional[_Union[_base_pb2.ModelProto, _Mapping]] = ...) -> None: ...

class ExperimentListItem(_message.Message):
    __slots__ = ["collaborators_amount", "name", "progress", "status", "tasks_amount"]
    COLLABORATORS_AMOUNT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASKS_AMOUNT_FIELD_NUMBER: _ClassVar[int]
    collaborators_amount: int
    name: str
    progress: float
    status: str
    tasks_amount: int
    def __init__(self, name: _Optional[str] = ..., status: _Optional[str] = ..., collaborators_amount: _Optional[int] = ..., tasks_amount: _Optional[int] = ..., progress: _Optional[float] = ...) -> None: ...

class GetDatasetInfoRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetDatasetInfoResponse(_message.Message):
    __slots__ = ["shard_info"]
    SHARD_INFO_FIELD_NUMBER: _ClassVar[int]
    shard_info: ShardInfo
    def __init__(self, shard_info: _Optional[_Union[ShardInfo, _Mapping]] = ...) -> None: ...

class GetEnvoysRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetEnvoysResponse(_message.Message):
    __slots__ = ["envoy_infos"]
    ENVOY_INFOS_FIELD_NUMBER: _ClassVar[int]
    envoy_infos: _containers.RepeatedCompositeFieldContainer[EnvoyInfo]
    def __init__(self, envoy_infos: _Optional[_Iterable[_Union[EnvoyInfo, _Mapping]]] = ...) -> None: ...

class GetExperimentDataRequest(_message.Message):
    __slots__ = ["collaborator_name", "experiment_name"]
    COLLABORATOR_NAME_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    collaborator_name: str
    experiment_name: str
    def __init__(self, experiment_name: _Optional[str] = ..., collaborator_name: _Optional[str] = ...) -> None: ...

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

class GetExperimentStatusRequest(_message.Message):
    __slots__ = ["experiment_name"]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_name: str
    def __init__(self, experiment_name: _Optional[str] = ...) -> None: ...

class GetExperimentStatusResponse(_message.Message):
    __slots__ = ["experiment_status"]
    EXPERIMENT_STATUS_FIELD_NUMBER: _ClassVar[int]
    experiment_status: str
    def __init__(self, experiment_status: _Optional[str] = ...) -> None: ...

class GetExperimentsListRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetExperimentsListResponse(_message.Message):
    __slots__ = ["experiments"]
    EXPERIMENTS_FIELD_NUMBER: _ClassVar[int]
    experiments: _containers.RepeatedCompositeFieldContainer[ExperimentListItem]
    def __init__(self, experiments: _Optional[_Iterable[_Union[ExperimentListItem, _Mapping]]] = ...) -> None: ...

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

class NodeInfo(_message.Message):
    __slots__ = ["cuda_devices", "name"]
    CUDA_DEVICES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    cuda_devices: _containers.RepeatedCompositeFieldContainer[CudaDeviceInfo]
    name: str
    def __init__(self, name: _Optional[str] = ..., cuda_devices: _Optional[_Iterable[_Union[CudaDeviceInfo, _Mapping]]] = ...) -> None: ...

class RemoveExperimentRequest(_message.Message):
    __slots__ = ["experiment_name"]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_name: str
    def __init__(self, experiment_name: _Optional[str] = ...) -> None: ...

class RemoveExperimentResponse(_message.Message):
    __slots__ = ["acknowledgement"]
    ACKNOWLEDGEMENT_FIELD_NUMBER: _ClassVar[int]
    acknowledgement: bool
    def __init__(self, acknowledgement: bool = ...) -> None: ...

class SetExperimentFailedRequest(_message.Message):
    __slots__ = ["collaborator_name", "error_code", "error_description", "experiment_name"]
    COLLABORATOR_NAME_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    collaborator_name: str
    error_code: int
    error_description: str
    experiment_name: str
    def __init__(self, experiment_name: _Optional[str] = ..., collaborator_name: _Optional[str] = ..., error_code: _Optional[int] = ..., error_description: _Optional[str] = ...) -> None: ...

class SetExperimentFailedResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class SetNewExperimentResponse(_message.Message):
    __slots__ = ["accepted"]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    def __init__(self, accepted: bool = ...) -> None: ...

class ShardInfo(_message.Message):
    __slots__ = ["n_samples", "node_info", "sample_shape", "shard_description", "target_shape"]
    NODE_INFO_FIELD_NUMBER: _ClassVar[int]
    N_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_SHAPE_FIELD_NUMBER: _ClassVar[int]
    SHARD_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TARGET_SHAPE_FIELD_NUMBER: _ClassVar[int]
    n_samples: int
    node_info: NodeInfo
    sample_shape: _containers.RepeatedScalarFieldContainer[str]
    shard_description: str
    target_shape: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, node_info: _Optional[_Union[NodeInfo, _Mapping]] = ..., shard_description: _Optional[str] = ..., n_samples: _Optional[int] = ..., sample_shape: _Optional[_Iterable[str]] = ..., target_shape: _Optional[_Iterable[str]] = ...) -> None: ...

class TrainedModelResponse(_message.Message):
    __slots__ = ["model_proto"]
    MODEL_PROTO_FIELD_NUMBER: _ClassVar[int]
    model_proto: _base_pb2.ModelProto
    def __init__(self, model_proto: _Optional[_Union[_base_pb2.ModelProto, _Mapping]] = ...) -> None: ...

class UpdateEnvoyStatusRequest(_message.Message):
    __slots__ = ["cuda_devices", "is_experiment_running", "name"]
    CUDA_DEVICES_FIELD_NUMBER: _ClassVar[int]
    IS_EXPERIMENT_RUNNING_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    cuda_devices: _containers.RepeatedCompositeFieldContainer[CudaDeviceInfo]
    is_experiment_running: bool
    name: str
    def __init__(self, name: _Optional[str] = ..., is_experiment_running: bool = ..., cuda_devices: _Optional[_Iterable[_Union[CudaDeviceInfo, _Mapping]]] = ...) -> None: ...

class UpdateEnvoyStatusResponse(_message.Message):
    __slots__ = ["health_check_period"]
    HEALTH_CHECK_PERIOD_FIELD_NUMBER: _ClassVar[int]
    health_check_period: _duration_pb2.Duration
    def __init__(self, health_check_period: _Optional[_Union[_duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class UpdateShardInfoRequest(_message.Message):
    __slots__ = ["shard_info"]
    SHARD_INFO_FIELD_NUMBER: _ClassVar[int]
    shard_info: ShardInfo
    def __init__(self, shard_info: _Optional[_Union[ShardInfo, _Mapping]] = ...) -> None: ...

class UpdateShardInfoResponse(_message.Message):
    __slots__ = ["accepted"]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    def __init__(self, accepted: bool = ...) -> None: ...

class WaitExperimentRequest(_message.Message):
    __slots__ = ["collaborator_name"]
    COLLABORATOR_NAME_FIELD_NUMBER: _ClassVar[int]
    collaborator_name: str
    def __init__(self, collaborator_name: _Optional[str] = ...) -> None: ...

class WaitExperimentResponse(_message.Message):
    __slots__ = ["experiment_name"]
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_name: str
    def __init__(self, experiment_name: _Optional[str] = ...) -> None: ...
