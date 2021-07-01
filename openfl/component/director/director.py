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

from openfl.federated import Plan
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import director_pb2
from openfl.protocols import director_pb2_grpc
from openfl.protocols.utils import construct_model_proto
from openfl.protocols.utils import deconstruct_model_proto

logger = logging.getLogger(__name__)


class Director(director_pb2_grpc.FederationDirectorServicer):
    """Director class."""

    def __init__(self, sample_shape: list, target_shape: list) -> None:
        """Initialize a director object."""
        # TODO: add working directory
        super().__init__()
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self.shard_registry = []
        self.experiments = set()
        self.col_exp_queues = defaultdict(asyncio.Queue)
        self.experiment_data = {}
        self.experiments_queue = asyncio.Queue()
        self.executor = ProcessPoolExecutor(max_workers=2)
        self.aggregator_task = None  # TODO: add check if exists and wait on terminate
        self.fqdn = socket.getfqdn()
        self.director_port = None
        self.tensorboard_port = 6006
        self.tensorboard_thread = None

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

        # TODO: save to file
        self.experiment_data[request.name] = npbytes
        tensor_dict = None
        if request.model_proto:
            tensor_dict, _ = deconstruct_model_proto(request.model_proto, NoCompressionPipeline())

        self.create_workspace(request.name, npbytes)
        self._run_aggregator(tensor_dict, request.name)

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
        if not hasattr(self, 'aggregator_server'):
            logger.error('Aggregator has not started yet')
            return director_pb2.TrainedModelResponse()
        elif self.aggregator_server.aggregator.last_tensor_dict is None:
            logger.error('Aggregator have no aggregated model to return')
            return director_pb2.TrainedModelResponse()

        if request.model_type == director_pb2.GetTrainedModelRequest.BEST_MODEL:
            tensor_dict = self.aggregator_server.aggregator.best_tensor_dict
        elif request.model_type == director_pb2.GetTrainedModelRequest.LAST_MODEL:
            tensor_dict = self.aggregator_server.aggregator.last_tensor_dict
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

    def _run_aggregator(
            self,
            initial_tensor_dict,
            experiment_name,
            plan='plan/plan.yaml',
    ):  # TODO: path params, change naming
        """Run aggregator."""
        cwd = os.getcwd()
        os.chdir(f'{cwd}/{experiment_name}')
        plan = Plan.parse(plan_config_path=Path(plan))
        plan.authorized_cols = list(self.col_exp_queues.keys())

        logger.info('🧿 Starting the Aggregator Service.')
        self.aggregator_server = plan.interactive_api_get_server(
            initial_tensor_dict,
            chain='../cert/root_ca.crt',
            certificate='../cert/dir_nnlicv674.inn.intel.com.crt',
            private_key='../cert/dir_nnlicv674.inn.intel.com.key')

        grpc_server = self.aggregator_server.get_server()
        grpc_server.start()

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
    


async def serve(*args, **kwargs):
    """Launch the director GRPC server."""
    from click import confirm
    from openfl.component.ca.ca import get_token 
    channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    server = aio.server(options=channel_opt)
    director = Director(*args, **kwargs)
    director_pb2_grpc.add_FederationDirectorServicer_to_server(director, server)
    director.run_tensorboard()
    # Add pass addr from director.yaml
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logger.info(f'Starting server on {listen_addr}')
    await server.start()
    await server.wait_for_termination()
