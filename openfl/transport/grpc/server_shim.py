# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AggregatorGRPCServer module."""

from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from multiprocessing import cpu_count
from time import sleep
from random import randint
from pathlib import Path

from grpc import server
from grpc import ssl_server_credentials

from openfl.protocols import Acknowledgement
from openfl.protocols import add_AggregatorServicer_to_server
from openfl.protocols import AggregatorServicer
from openfl.protocols import MessageHeader
from openfl.protocols import TaskResults
from openfl.protocols import TasksResponse
from openfl.protocols import TensorResponse
from openfl.protocols import utils
from openfl.utilities import check_equal
from openfl.utilities import check_is_in


class AggregatorGRPCServerShim(AggregatorServicer):
    """gRPC server class for the Aggregator."""

    def __init__(self,
                 aggregator,
                 agg_port,
                 tls=True,
                 disable_client_auth=False,
                 root_certificate=None,
                 certificate=None,
                 private_key=None,
                 proto_path='proto_path',
                 **kwargs):
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
        self.uri = f'[::]:{agg_port}'
        self.tls = tls
        self.disable_client_auth = disable_client_auth
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key
        self.proto_path = proto_path
        Path(self.proto_path).mkdir(parents=True, exist_ok=True)
        self.channel_options = [
            ('grpc.max_metadata_size', 32 * 1024 * 1024),
            ('grpc.max_send_message_length', 128 * 1024 * 1024),
            ('grpc.max_receive_message_length', 128 * 1024 * 1024)
        ]
        self.server = None
        self.server_credentials = None

        self.logger = getLogger(__name__)

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
            common_name = context.auth_context()[
                'x509_common_name'][0].decode('utf-8')
            collaborator_common_name = request.header.sender
            if not self.aggregator.valid_collaborator_cn_and_id(
                    common_name, collaborator_common_name):
                raise ValueError(
                    f'Invalid collaborator. CN: |{common_name}| '
                    f'collaborator_common_name: |{collaborator_common_name}|')

    def get_header(self, collaborator_name):
        """
        Compose and return MessageHeader.

        Args:
            collaborator_name : str
                The collaborator the message is intended for
        """
        return MessageHeader(
            sender=self.aggregator.uuid,
            receiver=collaborator_name,
            federation_uuid=self.aggregator.federation_uuid,
            single_col_cert_common_name=self.aggregator.single_col_cert_common_name
        )

    def check_request(self, request):
        """
        Validate request header matches expected values.

        Args:
            request : protobuf
                Request sent from a collaborator that requires validation
        """
        # TODO improve this check. the sender name could be spoofed
        check_is_in(request.header.sender, self.aggregator.authorized_cols, self.logger)

        # check that the message is for me
        check_equal(request.header.receiver, self.aggregator.uuid, self.logger)

        # check that the message is for my federation
        check_equal(
            request.header.federation_uuid, self.aggregator.federation_uuid, self.logger)

        # check that we agree on the single cert common name
        check_equal(
            request.header.single_col_cert_common_name,
            self.aggregator.single_col_cert_common_name,
            self.logger
        )

    def _generate_fnames(self, collaborator_name,func_name):
        """
        Generates filenames for request / response protobufs
        """
        request_id = randint(0,1000000000000)
        request = f'{self.proto_path}/{collaborator_name}_{func_name}_request_{request_id}.pbuf'
        response = f'{self.proto_path}/{collaborator_name}_{func_name}_response_{request_id}.pbuf'
        return request, response

    def _request_response_shim(self, request, collaborator_name, func_name):
        request_file, response_file = self._generate_fnames(collaborator_name,func_name)
        
        # Write to intermediate file to prevent premature request
        intermediate_req_file = request_file.replace('request','intermediate_req')
        with open(intermediate_req_file,'wb') as f:
            f.write(request.SerializeToString())
            self.logger.info(f'Serialized {func_name} request')
        
        # Request is now ready to be sent
        Path(intermediate_req_file).rename(request_file)

        while not Path(response_file).exists():
            self.logger.info(
                f'Waiting for {response_file} to become available.')
            sleep(1)

        if func_name == 'get_tasks':
            response = TasksResponse()
        elif func_name == 'get_aggregated_tensor':
            response = TensorResponse()
        elif func_name == 'send_local_task_results':
            return Acknowledgement(header=self.get_header(collaborator_name))
        else:
            self.logger.error(f'Bad function name {func_name} provided') 

        with open(response_file,'rb') as f:
            response.ParseFromString(f.read()) 

        # Remove response file
        Path(response_file).unlink()

        # Modify aggregator id
        response.header.sender = self.aggregator.uuid
        response.header.federation_uuid = self.aggregator.federation_uuid

        return response

    def GetTasks(self, request, context):  # NOQA:N802
        """
        Request a job from aggregator.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """

        collaborator_name = request.header.sender
        return self._request_response_shim(request,collaborator_name,'get_tasks')

    def GetAggregatedTensor(self, request, context):  # NOQA:N802
        """
        Request a job from aggregator.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        collaborator_name = request.header.sender
        return self._request_response_shim(request,collaborator_name,'get_aggregated_tensor')

    def SendLocalTaskResults(self, request, context):  # NOQA:N802
        """
        Request a model download from aggregator.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        proto = TaskResults()
        proto = utils.datastream_to_proto(proto, request)
        collaborator_name = proto.header.sender

        return self._request_response_shim(proto,collaborator_name,'send_local_task_results')

    def get_server(self):
        """Return gRPC server."""
        self.server = server(ThreadPoolExecutor(max_workers=cpu_count()),
                             options=self.channel_options)

        add_AggregatorServicer_to_server(self, self.server)

        if not self.tls:

            self.logger.warn(
                'gRPC is running on insecure channel with TLS disabled.')
            port = self.server.add_insecure_port(self.uri)
            self.logger.info(f'Insecure port: {port}')

        else:

            with open(self.private_key, 'rb') as f:
                private_key_b = f.read()
            with open(self.certificate, 'rb') as f:
                certificate_b = f.read()
            with open(self.root_certificate, 'rb') as f:
                root_certificate_b = f.read()

            if self.disable_client_auth:
                self.logger.warn('Client-side authentication is disabled.')

            self.server_credentials = ssl_server_credentials(
                ((private_key_b, certificate_b),),
                root_certificates=root_certificate_b,
                require_client_auth=not self.disable_client_auth
            )

            self.server.add_secure_port(self.uri, self.server_credentials)

        return self.server

    def serve(self):
        """Start an aggregator gRPC service."""
        self.get_server()

        self.logger.info('Starting Aggregator gRPC Server')
        self.server.start()

        try:
            while not self.aggregator.all_quit_jobs_sent():
                sleep(5)
        except KeyboardInterrupt:
            pass

        self.server.stop(0)
