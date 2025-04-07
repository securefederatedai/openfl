# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""AggregatorGRPCServer module."""

import logging
from random import random
from time import sleep

import grpc

from openfl.protocols import aggregator_pb2, aggregator_pb2_grpc, utils
from openfl.transport.grpc.common import create_grpc_server, create_header, synchronized

logger = logging.getLogger(__name__)


class AggregatorGRPCServer(aggregator_pb2_grpc.AggregatorServicer):
    """Aggregator gRPC Server.

    Attributes:
        aggregator (Aggregator): An instance of the Aggregator object that this server is serving.
        agg_port (int): Port to start gRPC server on.
        use_tls (bool): Whether to use TLS for the connection.
        require_client_auth (bool): Whether to enable client-side authentication, i.e. mTLS.
            Ignored if `use_tls=False`.
        root_certificate (str): The path to the root certificate for the TLS connection, ignored if
            `use_tls=False`.
        certificate (str): The path to the client's certificate for the TLS connection, ignored if
            `use_tls=False`.
        private_key (str): The path to the client's private key for the TLS connection, ignored if
            `use_tls=False`.
        root_certificate_refresher_cb (Callable): A callback function that receives no arguments and
            returns the current root certificate.
    """

    def __init__(
        self,
        aggregator,
        agg_port,
        use_tls=True,
        require_client_auth=True,
        root_certificate=None,
        certificate=None,
        private_key=None,
        root_certificate_refresher_cb=None,
        **kwargs,
    ):
        self.aggregator = aggregator
        self.uri = f"[::]:{agg_port}"
        self.use_tls = use_tls
        self.require_client_auth = require_client_auth
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key

        if hasattr(self.aggregator, "is_connector_available"):
            self.use_connector = self.aggregator.is_connector_available()
        else:
            self.use_connector = False

        if self.use_connector:
            self.interop_client = (
                self.aggregator.get_interop_client()
            )  # Initialize the interoperability client
        else:
            self.interop_client = None

        self.root_certificate_refresher_cb = root_certificate_refresher_cb

    def validate_collaborator(self, request, context):
        """Validate the collaborator.

        This method checks that the collaborator who sent the request is
        authorized to do so.

        Args:
            request (aggregator_pb2.MessageHeader): The request from the
                collaborator.
            context (grpc.ServicerContext): The context of the request.

        Raises:
            grpc.RpcError: If the collaborator or collaborator certificate is
                not authorized.
        """
        if self.use_tls:
            collaborator_common_name = request.header.sender
            if self.require_client_auth:
                common_name = context.auth_context()["x509_common_name"][0].decode("utf-8")
            else:
                common_name = collaborator_common_name

            if not self.aggregator.valid_collaborator_cn_and_id(
                common_name, collaborator_common_name
            ):
                # Random delay in authentication failures
                sleep(5 * random())  # nosec
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    f"Invalid collaborator. CN: |{common_name}| "
                    f"collaborator_common_name: |{collaborator_common_name}|",
                )

    def check_request(self, request):
        """Validate request header matches expected values.

        This method checks that the request is valid and was sent by an
            authorized collaborator.

        Args:
            request (aggregator_pb2.MessageHeader): Request sent from a
                collaborator that requires validation.

        Raises:
            ValueError: If the request is not valid.
        """
        assert request.header.sender in self.aggregator.authorized_cols, (
            f"Sender in request header is not authorized. "
            f"Expected: one of {self.aggregator.authorized_cols}, Actual: {request.header.sender}"
        )

        assert request.header.receiver == self.aggregator.uuid, (
            f"Receiver in request header does not match aggregator UUID. "
            f"Expected: {self.aggregator.uuid}, Actual: {request.header.receiver}"
        )

        assert request.header.federation_uuid == self.aggregator.federation_uuid, (
            f"Federation UUID in request header does not match. "
            f"Expected: {self.aggregator.federation_uuid}, Actual: {request.header.federation_uuid}"
        )

        assert (
            request.header.single_col_cert_common_name
            == self.aggregator.single_col_cert_common_name
        ), (
            f"Single collaborator certificate common name in request header does not match. "
            f"Expected: {self.aggregator.single_col_cert_common_name}, Actual: {request.header.single_col_cert_common_name}"  # noqa: E501
        )

    def Ping(self, request, context):  # NOQA:N802
        """Ping endpoint of the Aggregator server.

        This method handles a ping request from a collaborator.

        Args:
            request (aggregator_pb2.PingRequest): The ping request from the
                collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            aggregator_pb2.PingResponse: The response to the ping request.
        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        header = create_header(
            sender=self.aggregator.uuid,
            receiver=request.header.sender,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name,
        )

        return aggregator_pb2.PingResponse(header=header)

    def GetTasks(self, request, context):  # NOQA:N802
        """Request a job from aggregator.

        This method handles a request from a collaborator for a job.

        Args:
            request (aggregator_pb2.GetTasksRequest): The request from the
                collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            aggregator_pb2.GetTasksResponse: The response to the request.
        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender
        tasks, round_number, sleep_time, time_to_quit = self.aggregator.get_tasks(
            request.header.sender
        )
        if tasks:
            if isinstance(tasks[0], str):
                # backward compatibility
                tasks_proto = [
                    aggregator_pb2.Task(
                        name=task,
                    )
                    for task in tasks
                ]
            else:
                tasks_proto = [
                    aggregator_pb2.Task(
                        name=task.name,
                        function_name=task.function_name,
                        task_type=task.task_type,
                        apply_local=task.apply_local,
                    )
                    for task in tasks
                ]
        else:
            tasks_proto = []

        header = create_header(
            sender=self.aggregator.uuid,
            receiver=collaborator_name,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name,
        )

        return aggregator_pb2.GetTasksResponse(
            header=header,
            round_number=round_number,
            tasks=tasks_proto,
            sleep_time=sleep_time,
            quit=time_to_quit,
        )

    def GetAggregatedTensor(self, request, context):  # NOQA:N802
        """Request a job from aggregator.

        This method handles a request from a collaborator for an aggregated
        tensor.

        Args:
            request (aggregator_pb2.GetAggregatedTensorRequest): The request
                from the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            aggregator_pb2.GetAggregatedTensorResponse: The response to the
                request.
        """
        if self.use_connector:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "This method is not available in framework interopability mode.",
            )

        self.validate_collaborator(request, context)
        self.check_request(request)

        named_tensor = self.aggregator.get_aggregated_tensor(
            request.tensor_name,
            request.round_number,
            request.report,
            tuple(request.tags),
            request.require_lossless,
        )

        header = create_header(
            sender=self.aggregator.uuid,
            receiver=request.header.sender,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name,
        )

        return aggregator_pb2.GetAggregatedTensorResponse(
            header=header,
            round_number=request.round_number,
            tensor=named_tensor,
        )

    @synchronized
    def SendLocalTaskResults(self, request, context):  # NOQA:N802
        """Request a model download from aggregator.

        This method handles a request from a collaborator to send the results
        of a local task.

        Args:
            request (aggregator_pb2.SendLocalTaskResultsRequest): The request
                from the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            aggregator_pb2.SendLocalTaskResultsResponse: The response to the
                request.
        """
        try:
            proto = aggregator_pb2.TaskResults()
            proto = utils.datastream_to_proto(proto, request)
        except RuntimeError:
            raise RuntimeError(
                "Empty stream message, reestablishing connection from client to resume training..."
            )

        self.validate_collaborator(proto, context)
        # all messages get sanity checked
        self.check_request(proto)

        collaborator_name = proto.header.sender
        task_name = proto.task_name
        round_number = proto.round_number
        data_size = proto.data_size
        named_tensors = proto.tensors
        self.aggregator.send_local_task_results(
            collaborator_name, round_number, task_name, data_size, named_tensors
        )
        # turn data stream into local model update
        header = create_header(
            sender=self.aggregator.uuid,
            receiver=collaborator_name,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name,
        )

        return aggregator_pb2.SendLocalTaskResultsResponse(header=header)

    def InteropRelay(self, request, context):
        """
        Args:
            request (aggregator_pb2.InteropRelay): The request
                from the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            aggregator_pb2.InteropRelay: The response to the
            request.
        """
        if not self.use_connector:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "InteropRelay is only available in federated interopability mode.",
            )

        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender

        header = create_header(
            sender=self.aggregator.uuid,
            receiver=collaborator_name,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name,
        )

        # Forward the incoming OpenFL message to the local gRPC client
        return self.interop_client.send_receive(request, header=header)

    def serve(self):
        """Starts the aggregator gRPC server."""

        if self.use_connector:
            self.aggregator.start_connector()

        server = create_grpc_server(
            self.uri,
            self.use_tls,
            self.private_key,
            self.certificate,
            self.root_certificate,
            self.require_client_auth,
            self.root_certificate_refresher_cb,
        )
        aggregator_pb2_grpc.add_AggregatorServicer_to_server(self, server)

        logger.info("Starting Aggregator gRPC Server")
        server.start()

        while not self.aggregator.all_quit_jobs_sent():
            sleep(5)

        if self.use_connector:
            self.aggregator.stop_connector()

        server.stop(0)
