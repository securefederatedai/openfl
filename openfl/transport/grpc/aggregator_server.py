# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""AggregatorGRPCServer module."""

import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from random import random
from time import sleep

from grpc import StatusCode, server, ssl_server_credentials

from openfl.protocols import aggregator_pb2, aggregator_pb2_grpc, utils
from openfl.transport.grpc.grpc_channel_options import channel_options
from openfl.utilities import check_equal, check_is_in

logger = logging.getLogger(__name__)


class AggregatorGRPCServer(aggregator_pb2_grpc.AggregatorServicer):
    """GRPC server class for the Aggregator.

    This class implements a gRPC server for the Aggregator, allowing it to
    communicate with collaborators.

    Attributes:
        aggregator (Aggregator): The aggregator that this server is serving.
        uri (str): The URI that the server is serving on.
        tls (bool): Whether to use TLS for the connection.
        disable_client_auth (bool): Whether to disable client-side
            authentication.
        root_certificate (str): The path to the root certificate for the TLS
            connection.
        certificate (str): The path to the server's certificate for the TLS
            connection.
        private_key (str): The path to the server's private key for the TLS
            connection.
        server (grpc.Server): The gRPC server.
        server_credentials (grpc.ServerCredentials): The server's credentials.
    """

    def __init__(
        self,
        aggregator,
        agg_port,
        tls=True,
        disable_client_auth=False,
        root_certificate=None,
        certificate=None,
        private_key=None,
        **kwargs,
    ):
        """
        Initialize the AggregatorGRPCServer.

        Args:
            aggregator (Aggregator): The aggregator that this server is
                serving.
            agg_port (int): The port that the server is serving on.
            tls (bool): Whether to use TLS for the connection.
            disable_client_auth (bool): Whether to disable client-side
                authentication.
            root_certificate (str): The path to the root certificate for the
                TLS connection.
            certificate (str): The path to the server's certificate for the
                TLS connection.
            private_key (str): The path to the server's private key for the
                TLS connection.
            **kwargs: Additional keyword arguments.
        """
        self.aggregator = aggregator
        self.uri = f"[::]:{agg_port}"
        self.tls = tls
        self.disable_client_auth = disable_client_auth
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key
        self.server = None
        self.server_credentials = None

        self.logger = logging.getLogger(__name__)

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
        if self.tls:
            common_name = context.auth_context()["x509_common_name"][0].decode("utf-8")
            collaborator_common_name = request.header.sender
            if not self.aggregator.valid_collaborator_cn_and_id(
                common_name, collaborator_common_name
            ):
                # Random delay in authentication failures
                sleep(5 * random())  # nosec
                context.abort(
                    StatusCode.UNAUTHENTICATED,
                    f"Invalid collaborator. CN: |{common_name}| "
                    f"collaborator_common_name: |{collaborator_common_name}|",
                )

    def get_header(self, collaborator_name):
        """Compose and return MessageHeader.

        This method creates a MessageHeader for a message to the specified
        collaborator.

        Args:
            collaborator_name (str): The name of the collaborator to send the
                message to.

        Returns:
            aggregator_pb2.MessageHeader: The header for the message.
        """
        return aggregator_pb2.MessageHeader(
            sender=self.aggregator.uuid,
            receiver=collaborator_name,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name,
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
        # TODO improve this check. the sender name could be spoofed
        check_is_in(request.header.sender, self.aggregator.authorized_cols, self.logger)

        # check that the message is for me
        check_equal(request.header.receiver, self.aggregator.uuid, self.logger)

        # check that the message is for my federation
        check_equal(
            request.header.federation_uuid,
            self.aggregator.federation_uuid,
            self.logger,
        )

        # check that we agree on the single cert common name
        check_equal(
            request.header.single_col_cert_common_name,
            self.aggregator.single_col_cert_common_name,
            self.logger,
        )

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

        return aggregator_pb2.GetTasksResponse(
            header=self.get_header(collaborator_name),
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
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender
        tensor_name = request.tensor_name
        require_lossless = request.require_lossless
        round_number = request.round_number
        report = request.report
        tags = tuple(request.tags)

        named_tensor = self.aggregator.get_aggregated_tensor(
            collaborator_name,
            tensor_name,
            round_number,
            report,
            tags,
            require_lossless,
        )

        return aggregator_pb2.GetAggregatedTensorResponse(
            header=self.get_header(collaborator_name),
            round_number=round_number,
            tensor=named_tensor,
        )

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
        return aggregator_pb2.SendLocalTaskResultsResponse(
            header=self.get_header(collaborator_name)
        )

    def get_server(self):
        """
        Return gRPC server.

        This method creates a gRPC server if it does not already exist and
        returns it.

        Returns:
            grpc.Server: The gRPC server.
        """
        self.server = server(ThreadPoolExecutor(max_workers=cpu_count()), options=channel_options)

        aggregator_pb2_grpc.add_AggregatorServicer_to_server(self, self.server)

        if not self.tls:

            self.logger.warn("gRPC is running on insecure channel with TLS disabled.")
            port = self.server.add_insecure_port(self.uri)
            self.logger.info("Insecure port: %s", port)

        else:

            with open(self.private_key, "rb") as f:
                private_key_b = f.read()
            with open(self.certificate, "rb") as f:
                certificate_b = f.read()
            with open(self.root_certificate, "rb") as f:
                root_certificate_b = f.read()

            if self.disable_client_auth:
                self.logger.warn("Client-side authentication is disabled.")

            self.server_credentials = ssl_server_credentials(
                ((private_key_b, certificate_b),),
                root_certificates=root_certificate_b,
                require_client_auth=not self.disable_client_auth,
            )

            self.server.add_secure_port(self.uri, self.server_credentials)

        return self.server

    def serve(self):
        """Start an aggregator gRPC service.

        This method starts the gRPC server and handles requests until all quit
        jobs havebeen sent.
        """
        self.get_server()

        self.logger.info("Starting Aggregator gRPC Server")
        self.server.start()

        try:
            while not self.aggregator.all_quit_jobs_sent():
                sleep(5)
        except KeyboardInterrupt:
            pass

        self.server.stop(0)
