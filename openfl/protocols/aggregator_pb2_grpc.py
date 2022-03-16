# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from openfl.protocols import aggregator_pb2 as openfl_dot_protocols_dot_aggregator__pb2
from openfl.protocols import base_pb2 as openfl_dot_protocols_dot_base__pb2


class AggregatorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetTasks = channel.unary_unary(
                '/openfl.aggregator.Aggregator/GetTasks',
                request_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetTasksRequest.SerializeToString,
                response_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetTasksResponse.FromString,
                )
        self.GetAggregatedTensor = channel.unary_unary(
                '/openfl.aggregator.Aggregator/GetAggregatedTensor',
                request_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetAggregatedTensorRequest.SerializeToString,
                response_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetAggregatedTensorResponse.FromString,
                )
        self.SendLocalTaskResults = channel.stream_unary(
                '/openfl.aggregator.Aggregator/SendLocalTaskResults',
                request_serializer=openfl_dot_protocols_dot_base__pb2.DataStream.SerializeToString,
                response_deserializer=openfl_dot_protocols_dot_aggregator__pb2.SendLocalTaskResultsResponse.FromString,
                )
        self.GetMetricStream = channel.unary_stream(
                '/openfl.aggregator.Aggregator/GetMetricStream',
                request_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetMetricStreamRequest.SerializeToString,
                response_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetMetricStreamResponse.FromString,
                )
        self.GetTrainedModel = channel.unary_unary(
                '/openfl.aggregator.Aggregator/GetTrainedModel',
                request_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetTrainedModelRequest.SerializeToString,
                response_deserializer=openfl_dot_protocols_dot_aggregator__pb2.TrainedModelResponse.FromString,
                )
        self.GetExperimentDescription = channel.unary_unary(
                '/openfl.aggregator.Aggregator/GetExperimentDescription',
                request_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetExperimentDescriptionRequest.SerializeToString,
                response_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetExperimentDescriptionResponse.FromString,
                )
        self.Stop = channel.unary_unary(
                '/openfl.aggregator.Aggregator/Stop',
                request_serializer=openfl_dot_protocols_dot_aggregator__pb2.StopRequest.SerializeToString,
                response_deserializer=openfl_dot_protocols_dot_aggregator__pb2.StopResponse.FromString,
                )


class AggregatorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetTasks(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAggregatedTensor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendLocalTaskResults(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetMetricStream(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTrainedModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetExperimentDescription(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Stop(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_AggregatorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetTasks': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTasks,
                    request_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetTasksRequest.FromString,
                    response_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetTasksResponse.SerializeToString,
            ),
            'GetAggregatedTensor': grpc.unary_unary_rpc_method_handler(
                    servicer.GetAggregatedTensor,
                    request_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetAggregatedTensorRequest.FromString,
                    response_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetAggregatedTensorResponse.SerializeToString,
            ),
            'SendLocalTaskResults': grpc.stream_unary_rpc_method_handler(
                    servicer.SendLocalTaskResults,
                    request_deserializer=openfl_dot_protocols_dot_base__pb2.DataStream.FromString,
                    response_serializer=openfl_dot_protocols_dot_aggregator__pb2.SendLocalTaskResultsResponse.SerializeToString,
            ),
            'GetMetricStream': grpc.unary_stream_rpc_method_handler(
                    servicer.GetMetricStream,
                    request_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetMetricStreamRequest.FromString,
                    response_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetMetricStreamResponse.SerializeToString,
            ),
            'GetTrainedModel': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTrainedModel,
                    request_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetTrainedModelRequest.FromString,
                    response_serializer=openfl_dot_protocols_dot_aggregator__pb2.TrainedModelResponse.SerializeToString,
            ),
            'GetExperimentDescription': grpc.unary_unary_rpc_method_handler(
                    servicer.GetExperimentDescription,
                    request_deserializer=openfl_dot_protocols_dot_aggregator__pb2.GetExperimentDescriptionRequest.FromString,
                    response_serializer=openfl_dot_protocols_dot_aggregator__pb2.GetExperimentDescriptionResponse.SerializeToString,
            ),
            'Stop': grpc.unary_unary_rpc_method_handler(
                    servicer.Stop,
                    request_deserializer=openfl_dot_protocols_dot_aggregator__pb2.StopRequest.FromString,
                    response_serializer=openfl_dot_protocols_dot_aggregator__pb2.StopResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'openfl.aggregator.Aggregator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Aggregator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetTasks(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/openfl.aggregator.Aggregator/GetTasks',
            openfl_dot_protocols_dot_aggregator__pb2.GetTasksRequest.SerializeToString,
            openfl_dot_protocols_dot_aggregator__pb2.GetTasksResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAggregatedTensor(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/openfl.aggregator.Aggregator/GetAggregatedTensor',
            openfl_dot_protocols_dot_aggregator__pb2.GetAggregatedTensorRequest.SerializeToString,
            openfl_dot_protocols_dot_aggregator__pb2.GetAggregatedTensorResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendLocalTaskResults(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/openfl.aggregator.Aggregator/SendLocalTaskResults',
            openfl_dot_protocols_dot_base__pb2.DataStream.SerializeToString,
            openfl_dot_protocols_dot_aggregator__pb2.SendLocalTaskResultsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetMetricStream(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/openfl.aggregator.Aggregator/GetMetricStream',
            openfl_dot_protocols_dot_aggregator__pb2.GetMetricStreamRequest.SerializeToString,
            openfl_dot_protocols_dot_aggregator__pb2.GetMetricStreamResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTrainedModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/openfl.aggregator.Aggregator/GetTrainedModel',
            openfl_dot_protocols_dot_aggregator__pb2.GetTrainedModelRequest.SerializeToString,
            openfl_dot_protocols_dot_aggregator__pb2.TrainedModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetExperimentDescription(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/openfl.aggregator.Aggregator/GetExperimentDescription',
            openfl_dot_protocols_dot_aggregator__pb2.GetExperimentDescriptionRequest.SerializeToString,
            openfl_dot_protocols_dot_aggregator__pb2.GetExperimentDescriptionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Stop(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/openfl.aggregator.Aggregator/Stop',
            openfl_dot_protocols_dot_aggregator__pb2.StopRequest.SerializeToString,
            openfl_dot_protocols_dot_aggregator__pb2.StopResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
