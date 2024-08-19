# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""AggregatorGRPCClient module."""

import time
from logging import getLogger
from typing import Optional, Tuple

import grpc

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import aggregator_pb2, aggregator_pb2_grpc, utils
from openfl.transport.grpc.grpc_channel_options import channel_options
from openfl.utilities import check_equal


class ConstantBackoff:
    """Constant Backoff policy.

    This class implements a backoff policy that waits for a constant amount of
    time between retries.

    Attributes:
        reconnect_interval (int): The interval between connection attempts.
        logger (Logger): The logger to use for reporting connection attempts.
        uri (str): The URI to connect to.
    """

    def __init__(self, reconnect_interval, logger, uri):
        """Initialize Constant Backoff.

        Args:
            reconnect_interval (int): The interval between connection attempts.
            logger (Logger): The logger to use for reporting connection
                attempts.
            uri (str): The URI to connect to.
        """
        self.reconnect_interval = reconnect_interval
        self.logger = logger
        self.uri = uri

    def sleep(self):
        """Sleep for specified interval."""
        self.logger.info("Attempting to connect to aggregator at %s", self.uri)
        time.sleep(self.reconnect_interval)


class RetryOnRpcErrorClientInterceptor(
    grpc.UnaryUnaryClientInterceptor, grpc.StreamUnaryClientInterceptor
):
    """Retry gRPC connection on failure.

    This class implements a gRPC client interceptor that retries failed RPC
    calls.

    Attributes:
        sleeping_policy (ConstantBackoff): The backoff policy to use between
            retries.
        status_for_retry (Tuple[grpc.StatusCode]): The gRPC status codes that
            should trigger a retry.
    """

    def __init__(
        self,
        sleeping_policy,
        status_for_retry: Optional[Tuple[grpc.StatusCode]] = None,
    ):
        """Initialize function for gRPC retry.

        Args:
            sleeping_policy (ConstantBackoff): The backoff policy to use
                between retries.
            status_for_retry (Tuple[grpc.StatusCode], optional): The gRPC
                status codes that should trigger a retry.
        """
        self.sleeping_policy = sleeping_policy
        self.status_for_retry = status_for_retry

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        """Intercept the call to the gRPC server.

        Args:
            continuation (function): The original RPC call.
            client_call_details (grpc.ClientCallDetails): The details of the
                call.
            request_or_iterator (object): The request message for the RPC call.

        Returns:
            response (grpc.Call): The result of the RPC call.
        """
        while True:
            response = continuation(client_call_details, request_or_iterator)

            if isinstance(response, grpc.RpcError):

                # If status code is not in retryable status codes
                self.sleeping_policy.logger.info("Response code: %s", response.code())
                if self.status_for_retry and response.code() not in self.status_for_retry:
                    return response

                self.sleeping_policy.sleep()
            else:
                return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Wrap intercept call for unary->unary RPC.

        Args:
            continuation (function): The original RPC call.
            client_call_details (grpc.ClientCallDetails): The details of the
                call.
            request (object): The request message for the RPC call.

        Returns:
            grpc.Call: The result of the RPC call.
        """
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        """
        Wrap intercept call for stream->unary RPC.

        Args:
            continuation (function): The original RPC call.
            client_call_details (grpc.ClientCallDetails): The details of the
                call.
            request_iterator (iterator): The request messages for the RPC call.

        Returns:
            grpc.Call: The result of the RPC call.
        """
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
                        f"Attempting to resend data request to aggregator at {self.uri}"
                    )
                elif e.code() == grpc.StatusCode.UNAUTHENTICATED:
                    raise
                continue
            break
        return response

    return wrapper


class AggregatorGRPCClient:
    """Client to the aggregator over gRPC-TLS.

    This class implements a gRPC client for communicating with an aggregator
    over a secure (TLS) connection.

    Attributes:
        uri (str): The URI of the aggregator.
        tls (bool): Whether to use TLS for the connection.
        disable_client_auth (bool): Whether to disable client-side
            authentication.
        root_certificate (str): The path to the root certificate for the TLS
            connection.
        certificate (str): The path to the client's certificate for the TLS
            connection.
        private_key (str): The path to the client's private key for the TLS
            connection.
        aggregator_uuid (str): The UUID of the aggregator.
        federation_uuid (str): The UUID of the federation.
        single_col_cert_common_name (str): The common name on the
            collaborator's certificate.
    """

    def __init__(
        self,
        agg_addr,
        agg_port,
        disable_client_auth,
        root_certificate,
        certificate,
        private_key,
        tls=True,
        aggregator_uuid=None,
        federation_uuid=None,
        single_col_cert_common_name=None,
        **kwargs,
    ):
        """
        Initialize.

        Args:
            agg_addr (str): The address of the aggregator.
            agg_port (int): The port of the aggregator.
            tls (bool): Whether to use TLS for the connection.
            disable_client_auth (bool): Whether to disable client-side
                authentication.
            root_certificate (str): The path to the root certificate for the
                TLS connection.
            certificate (str): The path to the client's certificate for the
                TLS connection.
            private_key (str): The path to the client's private key for the
                TLS connection.
            aggregator_uuid (str,optional): The UUID of the aggregator.
            federation_uuid (str, optional): The UUID of the federation.
            single_col_cert_common_name (str, optional): The common name on
                the collaborator's certificate.
            **kwargs: Additional keyword arguments.
        """
        self.uri = f"{agg_addr}:{agg_port}"
        self.tls = tls
        self.disable_client_auth = disable_client_auth
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key

        self.logger = getLogger(__name__)

        if not self.tls:
            self.logger.warn("gRPC is running on insecure channel with TLS disabled.")
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.root_certificate,
                self.disable_client_auth,
                self.certificate,
                self.private_key,
            )

        self.header = None
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.single_col_cert_common_name = single_col_cert_common_name

        # Adding an interceptor for RPC Errors
        self.interceptors = (
            RetryOnRpcErrorClientInterceptor(
                sleeping_policy=ConstantBackoff(
                    logger=self.logger,
                    reconnect_interval=int(kwargs.get("client_reconnect_interval", 1)),
                    uri=self.uri,
                ),
                status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
            ),
        )
        self.stub = aggregator_pb2_grpc.AggregatorStub(
            grpc.intercept_channel(self.channel, *self.interceptors)
        )

    def create_insecure_channel(self, uri):
        """Set an insecure gRPC channel (i.e. no TLS) if desired.

        Warns user that this is not recommended.

        Args:
            uri (str): The uniform resource identifier for the insecure channel

        Returns:
            grpc.Channel: An insecure gRPC channel object
        """
        return grpc.insecure_channel(uri, options=channel_options)

    def create_tls_channel(
        self,
        uri,
        root_certificate,
        disable_client_auth,
        certificate,
        private_key,
    ):
        """
        Set an secure gRPC channel (i.e. TLS).

        Args:
            uri (str): The uniform resource identifier for the secure channel.
            root_certificate (str): The Certificate Authority filename.
            disable_client_auth (bool): True disables client-side
                authentication (not recommended, throws warning to user).
            certificate (str): The client certificate filename from the
                collaborator (signed by the certificate authority).
            private_key (str): The private key filename for the client
                certificate.

        Returns:
            grpc.Channel: A secure gRPC channel object
        """
        with open(root_certificate, "rb") as f:
            root_certificate_b = f.read()

        if disable_client_auth:
            self.logger.warn("Client-side authentication is disabled.")
            private_key_b = None
            certificate_b = None
        else:
            with open(private_key, "rb") as f:
                private_key_b = f.read()
            with open(certificate, "rb") as f:
                certificate_b = f.read()

        credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certificate_b,
            private_key=private_key_b,
            certificate_chain=certificate_b,
        )

        return grpc.secure_channel(uri, credentials, options=channel_options)

    def _set_header(self, collaborator_name):
        """Set the header for gRPC messages.

        Args:
            collaborator_name (str): The name of the collaborator.
        """
        self.header = aggregator_pb2.MessageHeader(
            sender=collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name or "",
        )

    def validate_response(self, reply, collaborator_name):
        """Validate the aggregator response.

        Args:
            reply (aggregator_pb2.MessageReply): The reply from the aggregator.
            collaborator_name (str): The name of the collaborator.
        """
        # check that the message was intended to go to this collaborator
        check_equal(reply.header.receiver, collaborator_name, self.logger)
        check_equal(reply.header.sender, self.aggregator_uuid, self.logger)

        # check that federation id matches
        check_equal(reply.header.federation_uuid, self.federation_uuid, self.logger)

        # check that there is aggrement on the single_col_cert_common_name
        check_equal(
            reply.header.single_col_cert_common_name,
            self.single_col_cert_common_name or "",
            self.logger,
        )

    def disconnect(self):
        """Close the gRPC channel."""
        self.logger.debug("Disconnecting from gRPC server at %s", self.uri)
        self.channel.close()

    def reconnect(self):
        """Create a new channel with the gRPC server."""
        # channel.close() is idempotent. Call again here in case it wasn't
        # issued previously
        self.disconnect()

        if not self.tls:
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.root_certificate,
                self.disable_client_auth,
                self.certificate,
                self.private_key,
            )

        self.logger.debug("Connecting to gRPC at %s", self.uri)

        self.stub = aggregator_pb2_grpc.AggregatorStub(
            grpc.intercept_channel(self.channel, *self.interceptors)
        )

    @_atomic_connection
    @_resend_data_on_reconnection
    def get_tasks(self, collaborator_name):
        """Get tasks from the aggregator.

        Args:
            collaborator_name (str): The name of the collaborator.

        Returns:
            Tuple[List[str], int, int, bool]: A tuple containing a list of
                tasks, the round number, the sleep time, and a boolean
                indicating whether to quit.
        """
        self._set_header(collaborator_name)
        request = aggregator_pb2.GetTasksRequest(header=self.header)
        response = self.stub.GetTasks(request)
        self.validate_response(response, collaborator_name)

        return (
            response.tasks,
            response.round_number,
            response.sleep_time,
            response.quit,
        )

    @_atomic_connection
    @_resend_data_on_reconnection
    def get_aggregated_tensor(
        self,
        collaborator_name,
        tensor_name,
        round_number,
        report,
        tags,
        require_lossless,
    ):
        """
        Get aggregated tensor from the aggregator.

        Args:
            collaborator_name (str): The name of the collaborator.
            tensor_name (str): The name of the tensor.
            round_number (int): The round number.
            report (str): The report.
            tags (List[str]): The tags.
            require_lossless (bool): Whether lossless compression is required.

        Returns:
            aggregator_pb2.TensorProto: The aggregated tensor.
        """
        self._set_header(collaborator_name)

        request = aggregator_pb2.GetAggregatedTensorRequest(
            header=self.header,
            tensor_name=tensor_name,
            round_number=round_number,
            report=report,
            tags=tags,
            require_lossless=require_lossless,
        )
        response = self.stub.GetAggregatedTensor(request)
        # also do other validation, like on the round_number
        self.validate_response(response, collaborator_name)

        return response.tensor

    @_atomic_connection
    @_resend_data_on_reconnection
    def send_local_task_results(
        self,
        collaborator_name,
        round_number,
        task_name,
        data_size,
        named_tensors,
    ):
        """
        Send task results to the aggregator.

        Args:
            collaborator_name (str): The name of the collaborator.
            round_number (int): The round number.
            task_name (str): The name of the task.
            data_size (int): The size of the data.
            named_tensors (List[aggregator_pb2.NamedTensorProto]): The list of
                named tensors.
        """
        self._set_header(collaborator_name)
        request = aggregator_pb2.TaskResults(
            header=self.header,
            round_number=round_number,
            task_name=task_name,
            data_size=data_size,
            tensors=named_tensors,
        )

        # convert (potentially) long list of tensors into stream
        stream = []
        stream += utils.proto_to_datastream(request, self.logger)
        response = self.stub.SendLocalTaskResults(iter(stream))

        # also do other validation, like on the round_number
        self.validate_response(response, collaborator_name)

    def _get_trained_model(self, experiment_name, model_type):
        """Get trained model RPC.

        Args:
            experiment_name (str): The name of the experiment.
            model_type (str): The type of the model.

        Returns:
            Dict[str, numpy.ndarray]: The trained model.
        """
        get_model_request = self.stub.GetTrainedModelRequest(
            experiment_name=experiment_name,
            model_type=model_type,
        )
        model_proto_response = self.stub.GetTrainedModel(get_model_request)
        tensor_dict, _ = utils.deconstruct_model_proto(
            model_proto_response.model_proto,
            NoCompressionPipeline(),
        )
        return tensor_dict
