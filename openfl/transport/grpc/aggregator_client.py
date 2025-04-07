# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""AggregatorGRPCClient module."""

import logging
import time
from typing import Optional, Tuple

import grpc

from openfl.protocols import aggregator_pb2, aggregator_pb2_grpc, utils
from openfl.transport.grpc.common import create_header, create_insecure_channel, create_tls_channel

logger = logging.getLogger(__name__)


class ConstantBackoff:
    """Constant Backoff policy.

    This class implements a backoff policy that waits for a constant amount of
    time between retries.

    Attributes:
        reconnect_interval (int): The interval between connection attempts.
        uri (str): The URI to connect to.
    """

    def __init__(self, reconnect_interval, uri):
        """Initialize Constant Backoff.

        Args:
            reconnect_interval (int): The interval between connection attempts.
            uri (str): The URI to connect to.
        """
        self.reconnect_interval = reconnect_interval
        self.uri = uri

    def sleep(self):
        """Sleep for specified interval."""
        logger.info("Attempting to connect to aggregator at %s", self.uri)
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
                logger.info("Response code: %s", response.code())
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
        if not self.enable_atomic_connections:
            return func(self, *args, **kwargs)
        self.reconnect()
        response = func(self, *args, **kwargs)
        self.disconnect()
        return response

    return wrapper


def _resend_data_on_reconnection(func):
    def wrapper(self, *args, **kwargs):
        if not self.resend_data_on_reconnection:
            return func(self, *args, **kwargs)
        while True:
            try:
                response = func(self, *args, **kwargs)
                break
            except grpc.RpcError as e:
                logger.info(
                    f"Failed to send data request to aggregator {self.uri}, error code {e.code()}"
                )
                if self.refetch_server_cert_callback is not None:
                    logger.info("Refetching server certificate")
                    self.root_certificate = self.refetch_server_cert_callback()
                if not self.enable_atomic_connections:
                    logger.info("Reconnecting to aggregator")
                    self.reconnect()
                self.sleeping_policy.sleep()
        return response

    return wrapper


class AggregatorGRPCClient:
    """Collaborator-side gRPC client that talks to the aggregator.

    Attributes:
        agg_addr (str): Aggregator address.
        agg_port (int): Aggregator port.
        aggregator_uuid (str): The UUID of the aggregator.
        federation_uuid (str): The UUID of the federation.
        collaborator_name (str): The common name of this collaborator.
        use_tls (bool): Whether to use TLS for the connection.
        require_client_auth (bool): Whether to enable client-side authentication, i.e. mTLS.
            Ignored if `use_tls=False`.
        root_certificate (str): The path to the root certificate for the TLS connection, ignored if
            `use_tls=False`.
        certificate (str): The path to the client's certificate for the TLS connection, ignored if
            `use_tls=False`.
        private_key (str): The path to the client's private key for the TLS connection, ignored if
            `use_tls=False`.
        single_col_cert_common_name (str): The common name on the
            collaborator's certificate.
        refetch_server_cert_callback (function): Callback function to refetch
            the server certificate.
        enable_atomic_connections (bool): Whether to use atomic connections (i.e. creates a new
            gRPC channel for each transaction and closes them immediately).
        resend_data_on_reconnection (bool): Whether to resend data on reconnection.
    """

    def __init__(
        self,
        agg_addr,
        agg_port,
        aggregator_uuid: str,
        federation_uuid: str,
        collaborator_name: str,
        use_tls=True,
        require_client_auth=True,
        root_certificate=None,
        certificate=None,
        private_key=None,
        single_col_cert_common_name=None,
        refetch_server_cert_callback=None,
        enable_atomic_connections=False,
        resend_data_on_reconnection=True,
        **kwargs,
    ):
        self.uri = f"{agg_addr}:{agg_port}"
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.collaborator_name = collaborator_name
        self.use_tls = use_tls
        self.require_client_auth = require_client_auth
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key
        self.single_col_cert_common_name = single_col_cert_common_name or ""
        self.refetch_server_cert_callback = refetch_server_cert_callback
        self.enable_atomic_connections = enable_atomic_connections
        self.resend_data_on_reconnection = resend_data_on_reconnection

        # Setup
        self.sleeping_policy = ConstantBackoff(
            reconnect_interval=kwargs.get("client_reconnect_interval", 1),
            uri=self.uri,
        )

        if not self.use_tls:
            logger.warning("gRPC is running on insecure channel with TLS disabled.")
            self.channel = create_insecure_channel(self.uri)
        else:
            self.channel = create_tls_channel(
                self.uri,
                self.root_certificate,
                self.require_client_auth,
                self.certificate,
                self.private_key,
            )

        self.stub = aggregator_pb2_grpc.AggregatorStub(self.channel)

    def validate_response(self, response):
        """Validate the aggregator response."""
        assert response.header.receiver == self.collaborator_name, (
            f"Receiver in response header does not match collaborator name. "
            f"Expected: {self.collaborator_name}, Actual: {response.header.receiver}"
        )
        assert response.header.sender == self.aggregator_uuid, (
            f"Sender in response header does not match aggregator UUID. "
            f"Expected: {self.aggregator_uuid}, Actual: {response.header.sender}"
        )
        assert response.header.federation_uuid == self.federation_uuid, (
            f"Federation UUID in response header does not match. "
            f"Expected: {self.federation_uuid}, Actual: {response.header.federation_uuid}"
        )
        assert response.header.single_col_cert_common_name == self.single_col_cert_common_name, (
            f"Single collaborator certificate common name in response header does not match. "
            f"Expected: {self.single_col_cert_common_name}, Actual: {response.header.single_col_cert_common_name}"  # noqa: E501
        )

    def disconnect(self):
        """Close the gRPC channel."""
        logger.info("Disconnecting from gRPC server at %s", self.uri)
        self.channel.close()

    def reconnect(self):
        """Create a new channel with the gRPC server."""
        # channel.close() is idempotent. Call again here in case it wasn't
        # issued previously
        self.disconnect()

        if not self.use_tls:
            self.channel = create_insecure_channel(self.uri)
        else:
            self.channel = create_tls_channel(
                self.uri,
                self.root_certificate,
                self.require_client_auth,
                self.certificate,
                self.private_key,
            )

        logger.info("Connecting to gRPC at %s", self.uri)

        self.stub = aggregator_pb2_grpc.AggregatorStub(self.channel)

    @_resend_data_on_reconnection
    @_atomic_connection
    def ping(self):
        """Ping the aggregator to check connectivity."""
        logger.info("Aggregator ping...")
        header = create_header(
            sender=self.collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name,
        )
        request = aggregator_pb2.PingRequest(header=header)
        response = self.stub.Ping(request)
        if response:
            self.validate_response(response)
            logger.info("Aggregator pong!")
        else:
            logger.warning("Aggregator ping failed...")

    @_resend_data_on_reconnection
    @_atomic_connection
    def get_tasks(self):
        """Get tasks from the aggregator.

        Args:
            collaborator_name (str): The name of the collaborator.

        Returns:
            Tuple[List[str], int, int, bool]: A tuple containing a list of
                tasks, the round number, the sleep time, and a boolean
                indicating whether to quit.
        """
        logger.info("Requesting tasks...")
        header = create_header(
            sender=self.collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name,
        )
        request = aggregator_pb2.GetTasksRequest(header=header)
        response = self.stub.GetTasks(request)
        self.validate_response(response)

        return (
            response.tasks,
            response.round_number,
            response.sleep_time,
            response.quit,
        )

    @_resend_data_on_reconnection
    @_atomic_connection
    def get_aggregated_tensor(
        self,
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
        header = create_header(
            sender=self.collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name,
        )

        request = aggregator_pb2.GetAggregatedTensorRequest(
            header=header,
            tensor_name=tensor_name,
            round_number=round_number,
            report=report,
            tags=tags,
            require_lossless=require_lossless,
        )
        response = self.stub.GetAggregatedTensor(request)
        self.validate_response(response)
        return response.tensor

    @_resend_data_on_reconnection
    @_atomic_connection
    def send_local_task_results(
        self,
        round_number,
        task_name,
        data_size=None,
        named_tensors=None,
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
        header = create_header(
            sender=self.collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name,
        )
        request = aggregator_pb2.TaskResults(
            header=header,
            round_number=round_number,
            task_name=task_name,
            data_size=data_size,
            tensors=named_tensors,
        )

        # convert (potentially) long list of tensors into stream
        response = self.stub.SendLocalTaskResults(utils.proto_to_datastream(request))
        self.validate_response(response)

    @_atomic_connection
    @_resend_data_on_reconnection
    def send_message_to_server(self, openfl_message, collaborator_name):
        """
        Forwards a converted message from the local GRPC server (LGS) to the OpenFL server and
        returns the response.

        Args:
            openfl_message: The converted message from the LGS to be sent to the OpenFL server.
            collaborator_name: The name of the collaborator.

        Returns:
            The response from the OpenFL server
        """
        header = create_header(
            sender=collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name,
        )
        openfl_message.header.CopyFrom(header)
        openfl_response = self.stub.InteropRelay(openfl_message)
        self.validate_response(openfl_response)
        return openfl_response
