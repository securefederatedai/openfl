# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director server."""

import asyncio
import logging
import uuid
from pathlib import Path

from grpc import aio
from grpc import ssl_server_credentials

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.protocols.utils import construct_model_proto
from openfl.protocols.utils import deconstruct_model_proto

logger = logging.getLogger(__name__)


class DirectorGRPCServer(director_pb2_grpc.FederationDirectorServicer):
    """Director transport class."""

    def __init__(self, *, director_cls, tls: bool = True,
                 root_certificate: str = None, private_key: str = None, certificate: str = None,
                 listen_host='[::]', listen_port=50051, **kwargs) -> None:
        """Initialize a director object."""
        # TODO: add working directory
        super().__init__()

        self.listen_uri = f'{listen_host}:{listen_port}'
        self.tls = tls
        self.root_certificate = None
        self.private_key = None
        self.certificate = None
        self._fill_certs(root_certificate, private_key, certificate)
        self.server = None
        self.director = director_cls(
            tls=self.tls,
            root_certificate=self.root_certificate,
            private_key=self.private_key,
            certificate=self.certificate,
            **kwargs
        )

    def _fill_certs(self, root_certificate, private_key, certificate):
        """Fill certificates."""
        if self.tls:
            if not (root_certificate and private_key and certificate):
                raise Exception('No certificates provided')
            self.root_certificate = Path(root_certificate).absolute()
            self.private_key = Path(private_key).absolute()
            self.certificate = Path(certificate).absolute()

    def validate_caller(self, request, context):
        """
        Validate the caller.

        Args:
            request: The gRPC message request
            context: The gRPC context

        Returns:
            True - if the caller has valid cert
            False - if the callers cert name is invalid
        """
        # Can we close the request right here?
        if self.tls:
            caller_cert_name = context.auth_context()['x509_common_name'][0].decode('utf-8')
            caller_common_name = request.header.sender
            if caller_cert_name != caller_common_name:
                return False

        return True

    def get_caller_name(self, context):
        caller_name = 'unauthorized_caller'
        if self.tls:
            caller_name = context.auth_context()['x509_common_name'][0].decode('utf-8')
        return caller_name

    def start(self):
        """Launch the director GRPC server."""
        asyncio.run(self._run())

    async def _run(self):
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        self.server = aio.server(options=channel_opt)
        director_pb2_grpc.add_FederationDirectorServicer_to_server(self, self.server)

        if not self.tls:
            self.server.add_insecure_port(self.listen_uri)
        else:
            with open(self.private_key, 'rb') as f:
                private_key_b = f.read()
            with open(self.certificate, 'rb') as f:
                certificate_b = f.read()
            with open(self.root_certificate, 'rb') as f:
                root_certificate_b = f.read()
            server_credentials = ssl_server_credentials(
                ((private_key_b, certificate_b),),
                root_certificates=root_certificate_b,
                require_client_auth=True
            )
            self.server.add_secure_port(self.listen_uri, server_credentials)
        logger.info(f'Starting server on {self.listen_uri}')
        await self.server.start()
        await self.server.wait_for_termination()

    async def AcknowledgeShard(self, shard_info, context):  # NOQA:N802
        """Receive acknowledge shard info."""
        logger.info(f'AcknowledgeShard request has got: {shard_info}')
        is_accepted = self.director.acknowledge_shard(shard_info)
        reply = director_pb2.ShardAcknowledgement(accepted=is_accepted)

        return reply

    async def SetNewExperiment(self, stream, context):  # NOQA:N802
        """Request to set new experiment."""
        logger.info(f'SetNewExperiment request has got {stream}')
        # TODO: add streaming reader
        data_file_path = Path(str(uuid.uuid4())).absolute()
        with open(data_file_path, 'wb') as data_file:
            async for request in stream:
                if request.experiment_data.size == len(request.experiment_data.npbytes):
                    data_file.write(request.experiment_data.npbytes)
                else:
                    raise Exception('Bad request')

        if not self.validate_caller(request, context):
            # Can we send reject before reading the stream?
            return director_pb2.SetNewExperimentResponse(accepted=False)
        tensor_dict = None
        if request.model_proto:
            tensor_dict, _ = deconstruct_model_proto(request.model_proto, NoCompressionPipeline())

        is_accepted = await self.director.set_new_experiment(
            experiment_name=request.name,
            sender_name=request.header.sender,
            tensor_dict=tensor_dict,
            collaborator_names=request.collaborator_names,
            data_file_path=data_file_path
        )

        logger.info('Send response')
        return director_pb2.SetNewExperimentResponse(accepted=is_accepted)

    async def GetTrainedModel(self, request, context):  # NOQA:N802
        """RPC for retrieving trained models."""
        if not self.validate_caller(request, context):
            return director_pb2.TrainedModelResponse()
        logger.info('Request GetTrainedModel has got!')

        if request.model_type == director_pb2.GetTrainedModelRequest.BEST_MODEL:
            model_type = 'best'
        elif request.model_type == director_pb2.GetTrainedModelRequest.LAST_MODEL:
            model_type = 'last'
        else:
            logger.error('Incorrect model type')
            return director_pb2.TrainedModelResponse()

        trained_model_dict = self.director.get_trained_model(
            experiment_name=request.experiment_name,
            caller=request.header.sender,
            model_type=model_type
        )

        if trained_model_dict is None:
            return director_pb2.TrainedModelResponse()

        model_proto = construct_model_proto(trained_model_dict, 0, NoCompressionPipeline())

        return director_pb2.TrainedModelResponse(model_proto=model_proto)

    async def GetExperimentData(self, request, context):  # NOQA:N802
        """Receive experiment data."""
        # TODO: add size filling
        # TODO: add experiment name field
        # TODO: rename npbytes to data
        data_file_path = self.director.get_experiment_data(request.experiment_name)
        max_buffer_size = (2 * 1024 * 1024)
        with open(data_file_path, 'rb') as df:
            while True:
                data = df.read(max_buffer_size)
                if len(data) == 0:
                    break
                yield director_pb2.ExperimentData(size=len(data), npbytes=data)

    async def WaitExperiment(self, request_iterator, context):  # NOQA:N802
        """Request for wait an experiment."""
        logger.info('Request WaitExperiment has got!')
        async for msg in request_iterator:
            logger.info(msg)
            experiment_name = await self.director.wait_experiment(msg.collaborator_name)
            logger.info(f'Experiment {experiment_name} was prepared')

            yield director_pb2.WaitExperimentResponse(experiment_name=experiment_name)

    async def GetDatasetInfo(self, request, context):  # NOQA:N802
        """Request the info about target and sample shapes in the dataset."""
        logger.info('Request GetDatasetInfo has got!')
        if not self.validate_caller(request, context):
            return director_pb2.ShardInfo()

        sample_shape, target_shape = self.director.get_dataset_info()
        resp = director_pb2.ShardInfo(
            sample_shape=sample_shape,
            target_shape=target_shape
        )
        return resp

    async def StreamMetrics(self, request, context):  # NOQA:N802
        """Request to stream metrics from the aggregator to frontend."""
        if not self.validate_caller(request, context):
            return

        logger.info(f'Request StreamMetrics for {request.experiment_name} experiment has got!')

        for metric_dict in self.director.stream_metrics(
                experiment_name=request.experiment_name, caller=request.header.sender
        ):
            if metric_dict is None:
                await asyncio.sleep(5)
                continue
            yield director_pb2.StreamMetricsResponse(**metric_dict)

    async def RemoveExperimentData(self, request, context):  # NOQA:N802
        """Remove experiment data RPC."""
        response = director_pb2.RemoveExperimentResponse(acknowledgement=False)
        if not self.validate_caller(request, context):
            return response

        self.director.remove_experiment_data(
            experiment_name=request.experiment_name,
            caller=request.header.sender)

        response.acknowledgement = True
        return response

    async def CollaboratorHealthCheck(self, request, context):  # NOQA:N802
        """Accept health check from envoy."""
        logger.debug(f'Request CollaboratorHealthCheck has got: {request}')
        health_check_period = self.director.collaborator_health_check(
            collaborator_name=request.name,
            is_experiment_running=request.is_experiment_running,
        )
        resp = director_pb2.CollaboratorHealthCheckResponse()
        resp.health_check_period.seconds = health_check_period

        return resp

    async def GetEnvoys(self, request, context):  # NOQA:N802
        """Get a status information about envoys."""
        envoy_infos = self.director.get_envoys()

        return director_pb2.GetEnvoysResponse(envoy_infos=envoy_infos)

    async def GetExperiments(self, request, context):  # NOQA:N802
        caller = self.get_caller_name(context)
        experiments = self.director.get_experiments(caller)
        experiment_descriptions_list = [
            director_pb2.ExperimentDescription(
                name=exp['name'],
                status=exp['status'],
                collaborators_amount=exp['collaborators_amount'],
                tasks_amount=exp['tasks_amount'],
                progress=exp['tasks_amount'],
            ) for exp in experiments
        ]
        return director_pb2.ExperimentDescriptionList(experiments=experiment_descriptions_list)
