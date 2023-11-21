# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AggregatorGRPCClient module."""

import asyncio
import logging
import time
from typing import Optional
from typing import Tuple

import grpc

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import aggregator_pb2
from openfl.protocols import aggregator_pb2_grpc
from openfl.protocols import utils
from openfl.utilities import check_equal

from .grpc_channel_options import channel_options

logger = logging.getLogger()

CLIENT_RECONNECT_INTERVAL = 1  # in seconds
CLIENT_CONNECT_TIMEOUT = 30  # in seconds


class ConstantBackoff:
    """Constant Backoff policy."""

    def __init__(self, reconnect_interval, uri):
        """Initialize Constant Backoff."""
        self.reconnect_interval = reconnect_interval
        self.uri = uri

    def sleep(self):
        """Sleep for specified interval."""
        logger.info(f'Attempting to connect to aggregator at {self.uri}')
        time.sleep(self.reconnect_interval)


class RetryOnRpcErrorClientInterceptor(
    grpc.UnaryUnaryClientInterceptor, grpc.StreamUnaryClientInterceptor
):
    """Retry gRPC connection on failure."""

    def __init__(
            self,
            sleeping_policy,
            status_for_retry: Optional[Tuple[grpc.StatusCode]] = None,
    ):
        """Initialize function for gRPC retry."""
        self.sleeping_policy = sleeping_policy
        self.status_for_retry = status_for_retry

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        """Intercept the call to the gRPC server."""
        while True:
            response = continuation(client_call_details, request_or_iterator)

            if isinstance(response, grpc.RpcError):

                # If status code is not in retryable status codes
                logger.info(f'Response code: {response.code()}')
                if (
                        self.status_for_retry
                        and response.code() not in self.status_for_retry
                ):
                    return response

                self.sleeping_policy.sleep()
            else:
                return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Wrap intercept call for unary->unary RPC."""
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(
            self, continuation, client_call_details, request_iterator
    ):
        """Wrap intercept call for stream->unary RPC."""
        return self._intercept_call(continuation, client_call_details, request_iterator)


class AsyncRetryOnRpcErrorClientInterceptorBase:
    """Async retry gRPC connection on failure."""

    def __init__(
            self,
            reconnect_interval=1,
            connect_timeout=30,
            status_for_retry: Optional[Tuple[grpc.StatusCode]] = None,
    ):
        """Initialize function for gRPC retry."""
        self.reconnect_interval = reconnect_interval
        self.connect_timeout = connect_timeout
        self.status_for_retry = status_for_retry

    async def handle_error_for_retry(self, error, started_time):
        """Suppress an exception and sleep if error is in status_for_retry."""
        logger.info(f'Response code: {error.code()}. Try to reconnect to the Aggregator.')
        if self.status_for_retry and error.code() not in self.status_for_retry:
            raise error
        if (time.time() - started_time) > self.connect_timeout:
            raise error
        await asyncio.sleep(self.reconnect_interval)


class AsyncRetryOnRpcErrorUnaryUnaryClientInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor, AsyncRetryOnRpcErrorClientInterceptorBase
):
    """Async retry gRPC connection on failure."""

    async def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        """Intercept the async call to the gRPC server."""
        started_time = time.time()
        while True:
            try:
                async_response = await continuation(client_call_details, request_or_iterator)
                response = await async_response
            except grpc.aio.AioRpcError as error:
                await self.handle_error_for_retry(error, started_time)
            else:
                return response

    async def intercept_unary_unary(self, continuation, client_call_details, request):
        """Wrap intercept call for unary->unary RPC."""
        return await self._intercept_call(continuation, client_call_details, request)


class AsyncRetryOnRpcErrorUnaryStreamClientInterceptor(
    grpc.aio.UnaryStreamClientInterceptor, AsyncRetryOnRpcErrorClientInterceptorBase
):
    """Async retry gRPC connection on failure."""

    async def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        """Intercept the async call to the gRPC server."""
        started_time = time.time()
        is_connected = False
        while True:
            try:
                async_iter = await continuation(client_call_details, request_or_iterator)
                async for response in async_iter:
                    yield response
                    is_connected = True
            except grpc.aio.AioRpcError as error:
                if (
                        error.code() == grpc.StatusCode.UNAVAILABLE
                        and error.details() == 'Socket closed'
                ) or is_connected:
                    logger.info(f'Rpc error: {error.code()}, details: {error.details()}.')
                    return
                await self.handle_error_for_retry(error, started_time)

    async def intercept_unary_stream(
            self, continuation, client_call_details, request_iterator
    ):
        """Wrap intercept call for unary->stream RPC."""
        return self._intercept_call(continuation, client_call_details, request_iterator)


def _atomic_connection(func):
    def wrapper(self, *args, **kwargs):
        self.reconnect()
        response = func(self, *args, **kwargs)
        self.disconnect()
        return response

    return wrapper


def _resend_data_on_reconnection(func):
    def wrapper(self, *args, **kwargs):
        while True:
            try:
                response = func(self, *args, **kwargs)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNKNOWN:
                    self.logger.info(
                        f'Attempting to resend data request to aggregator at {self.uri}'
                    )
                elif e.code() == grpc.StatusCode.UNAUTHENTICATED:
                    raise
                continue
            break
        return response

    return wrapper


class BaseAggregatorGRPCClient:
    """Base client to the aggregator over gRPC-TLS."""

    def __init__(self,
                 agg_addr,
                 agg_port,
                 tls,
                 disable_client_auth,
                 root_certificate,
                 certificate,
                 private_key,
                 aggregator_uuid=None,
                 federation_uuid=None,
                 single_col_cert_common_name=None,
                 **kwargs):
        """Initialize."""
        self.uri = f'{agg_addr}:{agg_port}'
        self.tls = tls
        self.disable_client_auth = disable_client_auth
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key
        self.kwargs = kwargs

        if not self.tls:
            logger.warn(
                'gRPC is running on insecure channel with TLS disabled.')
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.root_certificate,
                self.disable_client_auth,
                self.certificate,
                self.private_key
            )

        self.header = None
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.single_col_cert_common_name = single_col_cert_common_name
        self._create_stub(**kwargs)

    def _create_stub(self, **kwargs):
        """Create a server stub."""
        self.stub = aggregator_pb2_grpc.AggregatorStub(
            self.channel
        )

    def create_insecure_channel(self, uri):
        """
        Set an insecure gRPC channel (i.e. no TLS) if desired.

        Warns user that this is not recommended.

        Args:
            uri: The uniform resource identifier fo the insecure channel

        Returns:
            An insecure gRPC channel object

        """
        return grpc.insecure_channel(uri, options=channel_options)

    @staticmethod
    def get_credentials(root_certificate, disable_client_auth,
                        certificate, private_key):
        """
        Prepare credentials for an secure gRPC channel (i.e. TLS).

        Args:
            root_certificate: The Certificate Authority filename
            disable_client_auth (boolean): True disabled client-side
             authentication (not recommended, throws warning to user)
            certificate: The client certficate filename from the collaborator
             (signed by the certificate authority)

        Returns:
            A gRPC credentials object
        """
        with open(root_certificate, 'rb') as f:
            root_certificate_b = f.read()

        if disable_client_auth:
            logger.warn('Client-side authentication is disabled.')
            private_key_b = None
            certificate_b = None
        else:
            with open(private_key, 'rb') as f:
                private_key_b = f.read()
            with open(certificate, 'rb') as f:
                certificate_b = f.read()

        credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certificate_b,
            private_key=private_key_b,
            certificate_chain=certificate_b,
        )

        return credentials

    def create_tls_channel(self, uri, root_certificate, disable_client_auth,
                           certificate, private_key):
        """
        Set an secure gRPC channel (i.e. TLS).

        Args:
            uri: The uniform resource identifier fo the insecure channel
            root_certificate: The Certificate Authority filename
            disable_client_auth (boolean): True disabled client-side
             authentication (not recommended, throws warning to user)
            certificate: The client certficate filename from the collaborator
             (signed by the certificate authority)

        Returns:
            An insecure gRPC channel object
        """
        credentials = self.get_credentials(
            root_certificate,
            disable_client_auth,
            certificate,
            private_key
        )

        return grpc.secure_channel(
            uri, credentials, options=channel_options)

    def _set_header(self, collaborator_name):
        self.header = aggregator_pb2.MessageHeader(
            sender=collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name or ''
        )

    def validate_response(self, reply, collaborator_name):
        """Validate the aggregator response."""
        # check that the message was intended to go to this collaborator
        check_equal(reply.header.receiver, collaborator_name, logger)
        check_equal(reply.header.sender, self.aggregator_uuid, logger)

        # check that federation id matches
        check_equal(
            reply.header.federation_uuid,
            self.federation_uuid,
            logger
        )

        # check that there is aggrement on the single_col_cert_common_name
        check_equal(
            reply.header.single_col_cert_common_name,
            self.single_col_cert_common_name or '',
            logger
        )

    def disconnect(self):
        """Close the gRPC channel."""
        logger.debug(f'Disconnecting from gRPC server at {self.uri}')
        self.channel.close()

    def reconnect(self):
        """Create a new channel with the gRPC server."""
        # channel.close() is idempotent. Call again here in case it wasn't issued previously
        self.disconnect()

        if not self.tls:
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.root_certificate,
                self.disable_client_auth,
                self.certificate,
                self.private_key
            )

        logger.debug(f'Connecting to gRPC at {self.uri}')
        self._create_stub()


class AggregatorGRPCClient(BaseAggregatorGRPCClient):
    """Client to the aggregator over gRPC-TLS."""

    @property
    def interceptors(self):
        """Get interceptors."""
        interceptors = (
            RetryOnRpcErrorClientInterceptor(
                sleeping_policy=ConstantBackoff(
                    reconnect_interval=int(self.kwargs.get(
                        'client_reconnect_interval', CLIENT_RECONNECT_INTERVAL
                    )),
                    uri=self.uri),
                status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
            ),
        )
        return interceptors

    def _create_stub(self, **kwargs):
        """Create a server stub."""
        # Adding an interceptor for RPC Errors
        self.stub = aggregator_pb2_grpc.AggregatorStub(
            grpc.intercept_channel(self.channel, *self.interceptors)
        )

    @_atomic_connection
    @_resend_data_on_reconnection
    def get_tasks(self, collaborator_name):
        """Get tasks from the aggregator."""
        self._set_header(collaborator_name)
        request = aggregator_pb2.GetTasksRequest(header=self.header)
        response = self.stub.GetTasks(request)
        self.validate_response(response, collaborator_name)

        return response.tasks, response.round_number, response.sleep_time, response.quit

    @_atomic_connection
    @_resend_data_on_reconnection
    def get_aggregated_tensor(self, collaborator_name, tensor_name, round_number,
                              report, tags, require_lossless):
        """Get aggregated tensor from the aggregator."""
        self._set_header(collaborator_name)

        request = aggregator_pb2.GetAggregatedTensorRequest(
            header=self.header,
            tensor_name=tensor_name,
            round_number=round_number,
            report=report,
            tags=tags,
            require_lossless=require_lossless
        )
        response = self.stub.GetAggregatedTensor(request)
        # also do other validation, like on the round_number
        self.validate_response(response, collaborator_name)

        return response.tensor

    @_atomic_connection
    @_resend_data_on_reconnection
    def send_local_task_results(self, collaborator_name, round_number,
                                task_name, data_size, named_tensors):
        """Send task results to the aggregator."""
        self._set_header(collaborator_name)
        request = aggregator_pb2.TaskResults(
            header=self.header,
            round_number=round_number,
            task_name=task_name,
            data_size=data_size,
            tensors=named_tensors
        )

        # convert (potentially) long list of tensors into stream
        stream = []
        stream += utils.proto_to_datastream(request, logger)
        response = self.stub.SendLocalTaskResults(iter(stream))

        # also do other validation, like on the round_number
        self.validate_response(response, collaborator_name)


class AsyncAggregatorGRPCClient(BaseAggregatorGRPCClient):
    """Async client to the aggregator over gRPC-TLS."""

    @property
    def interceptors(self):
        """Get interceptors."""
        interceptors = (
            AsyncRetryOnRpcErrorUnaryUnaryClientInterceptor(
                reconnect_interval=int(self.kwargs.get(
                    'client_reconnect_interval', CLIENT_RECONNECT_INTERVAL
                )),
                connect_timeout=int(self.kwargs.get(
                    'client_connect_timeout', CLIENT_CONNECT_TIMEOUT
                )),
                status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
            ),
            AsyncRetryOnRpcErrorUnaryStreamClientInterceptor(
                reconnect_interval=int(self.kwargs.get(
                    'client_reconnect_interval', CLIENT_RECONNECT_INTERVAL
                )),
                connect_timeout=int(self.kwargs.get(
                    'client_connect_timeout', CLIENT_CONNECT_TIMEOUT
                )),
                status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
            ),
        )
        return interceptors

    def create_insecure_channel(self, uri):
        """
        Set an insecure gRPC channel (i.e. no TLS) if desired.

        Warns user that this is not recommended.

        Args:
            uri: The uniform resource identifier fo the insecure channel

        Returns:
            An insecure gRPC channel object

        """
        return grpc.aio.insecure_channel(
            uri,
            options=channel_options,
            interceptors=self.interceptors
        )

    def create_tls_channel(self, uri, root_certificate, disable_client_auth,
                           certificate, private_key):
        """
        Set an secure gRPC channel (i.e. TLS).

        Args:
            uri: The uniform resource identifier fo the insecure channel
            root_certificate: The Certificate Authority filename
            disable_client_auth (boolean): True disabled client-side
             authentication (not recommended, throws warning to user)
            certificate: The client certficate filename from the collaborator
             (signed by the certificate authority)

        Returns:
            An insecure gRPC channel object
        """
        credentials = self.get_credentials(
            root_certificate,
            disable_client_auth,
            certificate,
            private_key
        )

        return grpc.aio.secure_channel(
            uri, credentials, options=channel_options, interceptors=self.interceptors)

    async def get_trained_model(self, experiment_name, model_type):
        """Get trained model RPC."""
        get_model_request = aggregator_pb2.GetTrainedModelRequest(
            experiment_name=experiment_name,
            model_type=model_type,
        )
        model_proto_response = await self.stub.GetTrainedModel(get_model_request)
        tensor_dict, _ = utils.deconstruct_model_proto(
            model_proto_response.model_proto,
            NoCompressionPipeline(),
        )
        return tensor_dict

    async def get_metric_stream(self):
        """Get metrics stream."""
        request = aggregator_pb2.GetMetricStreamRequest()
        async for metric_message in self.stub.GetMetricStream(request):
            logger.debug(f'Message: {metric_message}')
            if not metric_message.metric_origin:
                yield None
                continue
            yield {
                'metric_origin': metric_message.metric_origin,
                'task_name': metric_message.task_name,
                'metric_name': metric_message.metric_name,
                'metric_value': metric_message.metric_value,
                'round': metric_message.round,
            }

    async def get_experiment_description(self):
        """Get experiment info."""
        request = aggregator_pb2.GetExperimentDescriptionRequest()
        response = await self.stub.GetExperimentDescription(request)
        return response.experiment
