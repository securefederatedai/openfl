# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""AggregatorGRPCServer module."""

import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from random import random
from time import sleep

from grpc import StatusCode, server, ssl_server_credentials

from openfl.experimental.protocols import aggregator_pb2, aggregator_pb2_grpc
from openfl.utilities import check_equal, check_is_in

from .grpc_channel_options import channel_options

logger = logging.getLogger(__name__)


class AggregatorGRPCServer(aggregator_pb2_grpc.AggregatorServicer):
    """gRPC server class for the Aggregator."""

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
        Class initializer.

        Args:
            aggregator: The aggregator
        Args:
            fltask (FLtask): The gRPC service task.
            tls (bool): To disable the TLS. (Default: True)
            disable_client_auth (bool): To disable the client side
            authentication. (Default: False)
            root_certificate (str): File path to the CA certificate.
            certificate (str): File path to the server certificate.
            private_key (str): File path to the private key.
            kwargs (dict): Additional arguments to pass into function
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
        """
        Validate the collaborator.

        Args:
            request: The gRPC message request
            context: The gRPC context

        Raises:
            ValueError: If the collaborator or collaborator certificate is not
             valid then raises error.

        """
        if self.tls:
            common_name = context.auth_context()["x509_common_name"][0].decode(
                "utf-8"
            )
            collaborator_common_name = request.header.sender
            if not self.aggregator.valid_collaborator_cn_and_id(
                common_name, collaborator_common_name
            ):
                # Random delay in authentication failures
                sleep(5 * random())
                context.abort(
                    StatusCode.UNAUTHENTICATED,
                    f"Invalid collaborator. CN: |{common_name}| "
                    f"collaborator_common_name: |{collaborator_common_name}|",
                )

    def get_header(self, collaborator_name):
        """
        Compose and return MessageHeader.

        Args:
            collaborator_name : str
                The collaborator the message is intended for
        """
        return aggregator_pb2.MessageHeader(
            sender=self.aggregator.uuid,
            receiver=collaborator_name,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name,
        )

    def check_request(self, request):
        """
        Validate request header matches expected values.

        Args:
            request : protobuf
                Request sent from a collaborator that requires validation
        """
        # TODO improve this check. the sender name could be spoofed
        check_is_in(
            request.header.sender, self.aggregator.authorized_cols, self.logger
        )

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

    def SendTaskResults(self, request, context):  # NOQA:N802
        """
        <FIND OUT WHAT COMMENT TO PUT HERE>.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender
        round_number = (request.round_number,)
        next_step = (request.next_step,)
        execution_environment = request.execution_environment

        _ = self.aggregator.send_task_results(
            collaborator_name, round_number[0], next_step, execution_environment
        )

        return aggregator_pb2.TaskResultsResponse(
            header=self.get_header(collaborator_name)
        )

    def GetTasks(self, request, context):  # NOQA:N802
        """
        Request a job from aggregator.

        Args:
            request: The gRPC message request
            context: The gRPC context
        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender

        rn, f, ee, st, q = self.aggregator.get_tasks(request.header.sender)

        return aggregator_pb2.GetTasksResponse(
            header=self.get_header(collaborator_name),
            round_number=rn,
            function_name=f,
            execution_environment=ee,
            sleep_time=st,
            quit=q,
        )

    def CallCheckpoint(self, request, context):  # NOQA:N802
        """
        Request aggregator to perform a checkpoint
        for a given function.

        Args:
            request: The gRPC message request
            context: The gRPC context
        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender
        execution_environment = request.execution_environment
        function = request.function
        stream_buffer = request.stream_buffer

        self.aggregator.call_checkpoint(
            execution_environment, function, stream_buffer
        )

        return aggregator_pb2.CheckpointResponse(
            header=self.get_header(collaborator_name)
        )

    def get_server(self):
        """Return gRPC server."""
        self.server = server(
            ThreadPoolExecutor(max_workers=cpu_count()), options=channel_options
        )

        aggregator_pb2_grpc.add_AggregatorServicer_to_server(self, self.server)

        if not self.tls:

            self.logger.warn(
                "gRPC is running on insecure channel with TLS disabled."
            )
            port = self.server.add_insecure_port(self.uri)
            self.logger.info(f"Insecure port: {port}")

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
        """Start an aggregator gRPC service."""
        self.get_server()

        self.logger.info("Starting Aggregator gRPC Server")
        self.server.start()
        self.is_server_started = True
        try:
            while not self.aggregator.all_quit_jobs_sent():
                sleep(5)
        except KeyboardInterrupt:
            pass

    def stop_server(self):
        self.server.stop(0)
