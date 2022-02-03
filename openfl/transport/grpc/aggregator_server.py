# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AggregatorGRPCServer module."""

import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from time import sleep

from grpc import server
from grpc import ssl_server_credentials

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import aggregator_pb2
from openfl.protocols import aggregator_pb2_grpc
from openfl.protocols import base_pb2
from openfl.protocols import utils
from openfl.utilities import check_equal
from openfl.utilities import check_is_in
from openfl.utilities.utils import getfqdn_env

logger = logging.getLogger(__name__)


class AggregatorGRPCServer(aggregator_pb2_grpc.AggregatorServicer):
    """gRPC server class for the Aggregator."""

    def __init__(self,
                 aggregator,
                 agg_port,
                 tls=True,
                 disable_client_auth=False,
                 root_certificate=None,
                 certificate=None,
                 private_key=None,
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
        self.channel_options = [
            ('grpc.max_metadata_size', 32 * 1024 * 1024),
            ('grpc.max_send_message_length', 128 * 1024 * 1024),
            ('grpc.max_receive_message_length', 128 * 1024 * 1024)
        ]
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
        return aggregator_pb2.MessageHeader(
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
        tasks, round_number, sleep_time, time_to_quit = self.aggregator.get_tasks(
            request.header.sender)

        return aggregator_pb2.GetTasksResponse(
            header=self.get_header(collaborator_name),
            round_number=round_number,
            tasks=tasks,
            sleep_time=sleep_time,
            quit=time_to_quit
        )

    def GetAggregatedTensor(self, request, context):  # NOQA:N802
        """
        Request a job from aggregator.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender
        tensor_name = request.tensor_name
        require_lossless = request.require_lossless
        round_number = request.round_number
        report = request.report
        tags = request.tags

        named_tensor = self.aggregator.get_aggregated_tensor(
            collaborator_name, tensor_name, round_number, report, tags, require_lossless)

        return aggregator_pb2.GetAggregatedTensorResponse(
            header=self.get_header(collaborator_name),
            round_number=round_number,
            tensor=named_tensor
        )

    def SendLocalTaskResults(self, request, context):  # NOQA:N802
        """
        Request a model download from aggregator.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        proto = aggregator_pb2.TaskResults()
        proto = utils.datastream_to_proto(proto, request)

        self.validate_collaborator(proto, context)
        # all messages get sanity checked
        self.check_request(proto)

        collaborator_name = proto.header.sender
        task_name = proto.task_name
        round_number = proto.round_number
        data_size = proto.data_size
        named_tensors = proto.tensors

        self.aggregator.send_local_task_results(
            collaborator_name, round_number, task_name, data_size, named_tensors)
        # turn data stream into local model update
        return aggregator_pb2.SendLocalTaskResultsResponse(
            header=self.get_header(collaborator_name)
        )

    def validate_director_request(self, context):
        """Validate if the request has got from the Director."""
        if self.tls:
            auth_subject = context.auth_context()['x509_common_name'][0].decode('utf-8')
            if auth_subject != getfqdn_env():
                raise Exception(f'Request has got from {auth_subject},'
                                f'but the Director FQDN is {getfqdn_env()}.')

    def GetMetricStream(self, request, context):  # NOQA:N802
        """Get metric stream."""
        self.validate_director_request(context)
        while True:
            try:
                logger.debug('Try to get metric dict')
                metric_dict = self.aggregator.metric_queue.get(timeout=1)
            except queue.Empty:
                if self.aggregator.all_quit_jobs_sent() and self.aggregator.metric_queue.empty():
                    return
                continue
            logger.debug('Metric dict has got. Yield it')
            yield aggregator_pb2.GetMetricStreamResponse(**metric_dict)

    def GetTrainedModel(self, request, context):  # NOQA:N802
        """RPC for retrieving trained models."""
        logger.info('Request GetTrainedModel has got.')
        self.validate_director_request(context)

        if request.model_type == aggregator_pb2.GetTrainedModelRequest.BEST_MODEL:
            model_type = 'best'
        elif request.model_type == aggregator_pb2.GetTrainedModelRequest.LAST_MODEL:
            model_type = 'last'
        else:
            logger.error('Incorrect model type')
            return aggregator_pb2.TrainedModelResponse()

        trained_model_dict = None
        if self.aggregator.last_tensor_dict is None:
            logger.error('Aggregator have no aggregated model to return')
        elif model_type == 'best':
            trained_model_dict = self.aggregator.best_tensor_dict
        elif model_type == 'last':
            trained_model_dict = self.aggregator.last_tensor_dict

        if trained_model_dict is None:
            return aggregator_pb2.TrainedModelResponse()

        model_proto = utils.construct_model_proto(
            trained_model_dict, 0, NoCompressionPipeline()
        )

        return aggregator_pb2.TrainedModelResponse(model_proto=model_proto)

    def GetExperimentDescription(self, request, context):  # NOQA:N802
        """Get an experiment description."""
        logger.info('GetExperimentDescription request has got.')
        self.validate_director_request(context)
        experiment = self.aggregator.get_experiment_description()
        models_statuses = [
            base_pb2.DownloadStatus(
                name=ms['name'],
                status=ms['status']
            )
            for ms in experiment['download_statuses']['models']
        ]
        logs_statuses = [
            base_pb2.DownloadStatus(
                name=ls['name'],
                status=ls['status']
            )
            for ls in experiment['download_statuses']['logs']
        ]
        download_statuses = base_pb2.DownloadStatuses(
            models=models_statuses,
            logs=logs_statuses,
        )
        collaborators = [
            base_pb2.CollaboratorDescription(
                name=col['name'],
                status=col['status'],
                progress=col['progress'],
                round=col['round'],
                current_task=col['current_task'],
                next_task=col['next_task']
            )
            for col in experiment['collaborators']
        ]
        tasks = [
            base_pb2.TaskDescription(
                name=task['name'],
                description=task['description']
            )
            for task in experiment['tasks']
        ]

        return aggregator_pb2.GetExperimentDescriptionResponse(
            experiment=base_pb2.ExperimentDescription(
                progress=experiment['progress'],
                current_round=experiment['current_round'],
                total_rounds=experiment['total_rounds'],
                download_statuses=download_statuses,
                collaborators=collaborators,
                tasks=tasks,
            ),
        )

    def get_server(self):
        """Return gRPC server."""
        self.server = server(ThreadPoolExecutor(max_workers=cpu_count() + 1),
                             options=self.channel_options)

        aggregator_pb2_grpc.add_AggregatorServicer_to_server(self, self.server)

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
