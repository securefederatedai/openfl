# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director module."""

import asyncio
import logging
import os
import shutil
import socket
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from grpc import aio
from grpc import ssl_server_credentials

from openfl.federated import Plan
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.protocols.utils import construct_model_proto
from openfl.protocols.utils import deconstruct_model_proto

logger = logging.getLogger(__name__)


class Director(director_pb2_grpc.FederationDirectorServicer):
    """Director class."""

    def __init__(self, disable_tls, root_ca, key, cert,
                 sample_shape: list, target_shape: list) -> None:
        """Initialize a director object."""
        # TODO: add working directory
        super().__init__()
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self.shard_registry = []
        self.col_exp_queues = defaultdict(asyncio.Queue)

        self.experiments = set()  # Do not know what it is
        self.experiment_data = {}  # {Experiment name : archive bytes}
        # What if two experimnets come with the same name from different users?
        self.experiments_queue = asyncio.Queue()  # Experimnets waiting to be executed
        self.experiment_stash = defaultdict(dict)  # Running of finished experimnets
        # {API name : {experiment name : aggregator}}

        self.executor = ProcessPoolExecutor(max_workers=2)
        self.aggregator_task = None  # TODO: add check if exists and wait on terminate
        self.fqdn = socket.getfqdn()
        self.director_port = None
        self.tensorboard_port = 6006
        self.tensorboard_thread = None
        self.disable_tls = disable_tls
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
        if not self.disable_tls:
            caller_cert_name = context.auth_context()[
                'x509_common_name'][0].decode('utf-8')
            caller_common_name = request.header.sender
            if not caller_cert_name == caller_common_name:
                return False

        return True

    async def AcknowledgeShard(self, shard_info, context):  # NOQA:N802
        """Receive acknowledge shard info."""
        logger.info(f'AcknowledgeShard request has got: {shard_info}')
        reply = director_pb2.ShardAcknowledgement(accepted=False)
        # If dataset do not match the data interface of the problem
        if (self.sample_shape != shard_info.sample_shape) or \
                (self.target_shape != shard_info.target_shape):
            logger.info('Request was not accepted')
            return reply
        logger.info('Request was accepted')
        self.shard_registry.append(shard_info)

        reply.accepted = True
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
            return director_pb2.SetNewExperimentResponse(
                accepted=False)

        # TODO: save to file
        self.experiment_data[request.name] = npbytes
        tensor_dict = None
        if request.model_proto:
            tensor_dict, _ = deconstruct_model_proto(request.model_proto, NoCompressionPipeline())

        self.create_workspace(request.name, npbytes)
        asyncio.create_task(self._run_aggregator(
            request.header.sender,
            tensor_dict,
            request.name,
            request.collaborator_names))

        logger.info(f'New experiment {request.name} for '
                    f'collaborators {request.collaborator_names}')
        for col_name in request.collaborator_names:
            queue = self.col_exp_queues[col_name]
            await queue.put(request.name)
        logger.info('Send response')
        return director_pb2.SetNewExperimentResponse(
            accepted=True,
            tensorboard_address=f'http://{self.fqdn}:{self.tensorboard_port}'
        )

    async def GetTrainedModel(self, request, context):  # NOQA:N802
        """RPC for retrieving trained models."""
        if not self.validate_caller(request, context):
            return director_pb2.TrainedModelResponse()
        logger.info('Request GetTrainedModel has got!')

        caller = request.header.sender
        exp_name = request.experiment_name

        if exp_name not in self.experiment_stash[caller]:
            logger.error('Aggregator has not started yet')
            return director_pb2.TrainedModelResponse()

        aggregator = self.experiment_stash[caller][exp_name].aggregator

        if aggregator.last_tensor_dict is None:
            logger.error('Aggregator have no aggregated model to return')
            return director_pb2.TrainedModelResponse()

        if request.model_type == director_pb2.GetTrainedModelRequest.BEST_MODEL:
            tensor_dict = aggregator.best_tensor_dict
        elif request.model_type == director_pb2.GetTrainedModelRequest.LAST_MODEL:
            tensor_dict = aggregator.last_tensor_dict
        else:
            logger.error('Incorrect model type')
            return director_pb2.TrainedModelResponse()

        model_proto = construct_model_proto(tensor_dict, 0, NoCompressionPipeline())

        return director_pb2.TrainedModelResponse(model_proto=model_proto)

    async def GetExperimentData(self, request, context):  # NOQA:N802
        """Receive experiment data."""
        # TODO: add size filling
        # TODO: add experiment name field
        # TODO: rename npbytes to data
        content = self.experiment_data.get(request.experiment_name, b'')
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
        queue = self.col_exp_queues[msg.collaborator_name]
        experiment_name = await queue.get()
        logger.info(f'Experiment {experiment_name} was prepared')

        yield director_pb2.WaitExperimentResponse(experiment_name=experiment_name)

    async def GetShardsInfo(self, request, context):  # NOQA:N802
        """Request a shard info."""
        logger.info('Request GetShardsInfo has got!')
        if not self.validate_caller(request, context):
            return director_pb2.ShardInfo()
        resp = director_pb2.ShardInfo(
            sample_shape=self.sample_shape,
            target_shape=self.target_shape
        )
        return resp

    async def GetRegisterdShards(self, request, context):  # NOQA:N802
        """Request registered shards."""
        logger.info('Request GetRegisterdShards has got!')
        resp = director_pb2.GetRegisterdShardsResponse(
            shard_info=self.shard_registry
        )
        return resp

    async def StreamMetrics(self, request, context):  # NOQA:N802
        """Request to stream metrics from the aggregator to frontend."""
        if not self.validate_caller(request, context):
            return
        caller = request.header.sender
        exp_name = request.experiment_name
        # We should probably set a name to the aggregator and verify it here.
        # Moreover, we may save the experiment name in plan.yaml and retrieve it
        # during the aggregator initialization
        logger.info(f'Request StreamMetrics for {exp_name} experimnet has got!')

        aggregator = self.experiment_stash[caller][exp_name].aggregator

        while not aggregator.all_quit_jobs_sent() or \
                not aggregator.metric_queue.empty():
            # If the aggregator has not fineished the experimnet
            # or it finished but metric_queue is not empty we send metrics

            # But here we may have a problem if the new experiment starts too quickly

            while not aggregator.metric_queue.empty():
                metric_origin, task_name, metric_name, metric_value, round_ = \
                    aggregator.metric_queue.get()
                yield director_pb2.StreamMetricsResponse(
                    metric_origin=metric_origin,
                    task_name=task_name,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    round=round_)

            # Awaiting quit job sent to collaborators
            await asyncio.sleep(5)

    async def RemoveExperimentData(self, request, context):
        """Remove experiment data RPC."""
        response = director_pb2.RemoveExperimnetResponse(acknowledgement=False)
        if not self.validate_caller(request, context):
            return response

        caller = request.header.sender
        if request.experiment_name in self.experiment_stash[caller]:
            del self.experiment_stash[caller][request.experiment_name]

        response.acknowledgement = True
        return response

    @staticmethod
    def create_workspace(experiment_name, npbytes):
        """Create the aggregator workspace."""
        if os.path.exists(experiment_name):
            shutil.rmtree(experiment_name)
        os.makedirs(experiment_name)

        arch_name = f'{experiment_name}/{experiment_name}' + '.zip'
        logger.info(f'arch_name: {arch_name}')
        with open(arch_name, 'wb') as content_file:
            content_file.write(npbytes)

        shutil.unpack_archive(arch_name, experiment_name)

    async def _run_aggregator(
            self,
            experiment_sender,
            initial_tensor_dict,
            experiment_name,
            collaborator_names,
            plan='plan/plan.yaml',
    ):  # TODO: path params, change naming
        """Run aggregator."""
        cwd = os.getcwd()
        os.chdir(f'{cwd}/{experiment_name}')
        plan = Plan.parse(plan_config_path=Path(plan))

        plan.authorized_cols = list(collaborator_names)

        logger.info('🧿 Starting the Aggregator Service.')
        aggregator_server = plan.interactive_api_get_server(
            initial_tensor_dict,
            chain=self.root_ca,
            certificate=self.cert,
            private_key=self.key
        )
        self.experiment_stash[experiment_sender][experiment_name] = \
            aggregator_server

        grpc_server = aggregator_server.get_server()
        grpc_server.start()
        logger.info('Starting Aggregator gRPC Server')

        try:
            while not aggregator_server.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            os.chdir(cwd)
            shutil.rmtree(experiment_name)
            grpc_server.stop(0)
            # Temporary solution to free RAM used by TensorDB
            aggregator_server.aggregator.tensor_db.clean_up(0)

    def run_tensorboard(self):
        """Run the tensorboard."""
        log_path = os.getcwd()
        self.tensorboard_thread = threading.Thread(
            target=lambda: os.system(
                f'tensorboard --logdir={log_path} --host={"0.0.0.0"} '
                f'--port={self.tensorboard_port}'
            ),
        )
        try:
            self.tensorboard_thread.start()
        except Exception as exc:
            logger.error(f'Failed to run tensorboard: {exc}')


async def serve(*args, disable_tls=False, root_ca=None, key=None, cert=None, **kwargs):
    """Launch the director GRPC server."""
    channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    server = aio.server(options=channel_opt)
    director = Director(*args, disable_tls, root_ca, key, cert, **kwargs)
    director_pb2_grpc.add_FederationDirectorServicer_to_server(director, server)
    director.run_tensorboard()
    # Add pass addr from director.yaml
    listen_addr = '[::]:50051'
    if disable_tls:
        server.add_insecure_port(listen_addr)
    else:
        if not (root_ca and key and cert):
            raise Exception('No certificates provided')
        with open(key, 'rb') as f:
            key_b = f.read()
        with open(cert, 'rb') as f:
            cert_b = f.read()
        with open(root_ca, 'rb') as f:
            root_ca_b = f.read()
        server_credentials = ssl_server_credentials(
            ((key_b, cert_b),),
            root_certificates=root_ca_b,
            require_client_auth=True
        )
        server.add_secure_port(listen_addr, server_credentials)
    logger.info(f'Starting server on {listen_addr}')
    await server.start()
    await server.wait_for_termination()
