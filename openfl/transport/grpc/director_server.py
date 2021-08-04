# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director server."""

import asyncio
import logging
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
                 root_ca: str = None, key: str = None, cert: str = None,
                 listen_ip='[::]', listen_port=50051, **kwargs) -> None:
        """Initialize a director object."""
        # TODO: add working directory
        super().__init__()

        self.listen_addr = f'{listen_ip}:{listen_port}'
        self.tls = tls
        self.root_ca = None
        self.key = None
        self.cert = None
        self._fill_certs(root_ca, key, cert)
        self.server = None
        self.director = director_cls(
            tls=self.tls,
            root_ca=self.root_ca,
            key=self.key,
            cert=self.cert,
            **kwargs
        )

    def _fill_certs(self, root_ca, key, cert):
        """Fill certificates."""
        if self.tls:
            if not (root_ca and key and cert):
                raise Exception('No certificates provided')
            self.root_ca = Path(root_ca).absolute()
            self.key = Path(key).absolute()
            self.cert = Path(cert).absolute()

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

    def start(self):
        """Launch the director GRPC server."""
        asyncio.run(self._run())

    async def _run(self):
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        self.server = aio.server(options=channel_opt)
        director_pb2_grpc.add_FederationDirectorServicer_to_server(self, self.server)

        if not self.tls:
            self.server.add_insecure_port(self.listen_addr)
        else:
            with open(self.key, 'rb') as f:
                key_b = f.read()
            with open(self.cert, 'rb') as f:
                cert_b = f.read()
            with open(self.root_ca, 'rb') as f:
                root_ca_b = f.read()
            server_credentials = ssl_server_credentials(
                ((key_b, cert_b),),
                root_certificates=root_ca_b,
                require_client_auth=True
            )
            self.server.add_secure_port(self.listen_addr, server_credentials)
        logger.info(f'Starting server on {self.listen_addr}')
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
        npbytes = b''
        async for request in stream:
            if request.experiment_data.size == len(request.experiment_data.npbytes):
                npbytes += request.experiment_data.npbytes
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
            data=npbytes
        )

        logger.info('Send response')
        return director_pb2.SetNewExperimentResponse(accepted=is_accepted)

    async def GetTrainedModel(self, request, context):  # NOQA:N802
        """RPC for retrieving trained models."""
        if not self.validate_caller(request, context):
            return director_pb2.TrainedModelResponse()
        logger.info('Request GetTrainedModel has got!')

        best_tensor_dict, last_tensor_dict = await self.director.get_trained_model(
            request.experiment_name,
            request.header.sender
        )

        if request.model_type == director_pb2.GetTrainedModelRequest.BEST_MODEL:
            tensor_dict = best_tensor_dict
        elif request.model_type == director_pb2.GetTrainedModelRequest.LAST_MODEL:
            tensor_dict = last_tensor_dict
        else:
            logger.error('Incorrect model type')
            return director_pb2.TrainedModelResponse()
        if not tensor_dict:
            return director_pb2.TrainedModelResponse()

        model_proto = construct_model_proto(tensor_dict, 0, NoCompressionPipeline())

        return director_pb2.TrainedModelResponse(model_proto=model_proto)

    async def GetExperimentData(self, request, context):  # NOQA:N802
        """Receive experiment data."""
        # TODO: add size filling
        # TODO: add experiment name field
        # TODO: rename npbytes to data
        content = self.director.get_experiment_data(request.experiment_name)
        logger.info(f'Content length: {len(content)}')
        max_buffer_size = (2 * 1024 * 1024)

        for i in range(0, len(content), max_buffer_size):
            chunk = content[i:i + max_buffer_size]
            logger.info(f'Send {len(chunk)} bytes')
            yield director_pb2.ExperimentData(size=len(chunk), npbytes=chunk)

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
        # We should probably set a name to the aggregator and verify it here.
        # Moreover, we may save the experiment name in plan.yaml and retrieve it
        # during the aggregator initialization
        logger.info(f'Request StreamMetrics for {request.experiment_name} experiment has got!')
        metrics = self.director.stream_metrics(request.experiment_name, request.header.sender)
        async for message in metrics:
            yield message

    async def RemoveExperimentData(self, request, context):  # NOQA:N802
        """Remove experiment data RPC."""
        response = director_pb2.RemoveExperimentResponse(acknowledgement=False)
        if not self.validate_caller(request, context):
            return response

        self.director.remove_experiment_data(request.experiment_name, request.header.sender)

        response.acknowledgement = True
        return response

    async def CollaboratorHealthCheck(self, request, context):  # NOQA:N802
        """Accept health check from envoy."""
        logger.debug(f'Request CollaboratorHealthCheck has got: {request}')
        is_accepted = self.director.collaborator_health_check(
            collaborator_name=request.name,
            is_experiment_running=request.is_experiment_running,
            valid_duration=request.valid_duration.seconds,
        )

        return director_pb2.CollaboratorHealthCheckResponse(accepted=is_accepted)

    async def GetEnvoys(self, request, context):  # NOQA:N802
        """Get a status information about envoys."""
        envoy_infos = self.director.get_envoys()

        return director_pb2.GetEnvoysResponse(envoy_infos=envoy_infos)
